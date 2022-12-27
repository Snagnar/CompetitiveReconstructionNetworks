from torchvision import transforms
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score
from pathlib import Path
import logging
import math
from tqdm import tqdm
import gc
import torch
from torch.nn import ModuleList
from typing import List
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid, save_image
import time

from training.networks import create_optimizer, create_scheduler
from training.skipnetworks import UnetGenerator, UnetGeneratorImproved, get_norm_layer


ADD_ADDITIONAL_UNITS = False

class CompetitiveReconstructionNetwork(LightningModule):
    def __init__(
            self,
            input_shape,
            image_output_interval: int,
            k_lambda: float,
            optimizer: str,
            momentum: float,
            lr: float,
            lr_scheduler: str,
            lr_factor: float,
            lr_steps: List[int],
            epochs: int,
            image_output_path: str,
            num_competitive_units: int,
            self_discrimination: bool,
            bagging: bool,
            feedback_loss_reduction: str,
            discrimination_loss_reduction: str,
            ignore_feedback: bool,
            network_depth: int,
            dynamic_loss_weights: bool,
            minimum_learning_units: int,
            feedback_weight: float,
            reconstruction_weight: float,
            discrimination_weight: float,
            conv_bias: bool,
            stride: int, 
            use_dropout: bool,
            feedback_group_size: int = -1,
            norm: str = "instance",
            improved: bool = False,
            **kwargs):
        super(CompetitiveReconstructionNetwork, self).__init__()
        self.save_hyperparameters()
        
        self.automatic_optimization = False
        if image_output_path is not None:
            self.image_output_path = Path(image_output_path)
            self.image_output_path.mkdir(parents=True, exist_ok=True)
            # for output_file in self.image_output_path.iterdir():
            #     output_file.unlink()
        else:
            self.image_output_path = None
        
        self.train_start_time = time.time()

        if wandb.run is not None:
            wandb.config.update({"network_depth": network_depth}, allow_val_change=True)
        network_depth = int(math.log2(input_shape[2]))
        logging.info(f"using network depth {network_depth}")
        if improved:
            self.competitive_units = [UnetGeneratorImproved(input_shape[1], input_shape[1], network_depth, norm_layer=get_norm_layer(norm), use_dropout=use_dropout).to(self.device) for _ in range(num_competitive_units)]
        else:
            self.competitive_units = [UnetGenerator(input_shape[1], input_shape[1], network_depth, norm_layer=get_norm_layer(norm), use_dropout=use_dropout).to(self.device) for _ in range(num_competitive_units)]
        self.convergences = [0] * len(self.competitive_units)
        self.generator_losses = [0] * len(self.competitive_units)
        self.discriminator_losses = [0] * len(self.competitive_units)
        self.feedback_losses = [[0 for _ in range(len(self.competitive_units))] for _ in range(len(self.competitive_units))]
        self.recon_feedback_diff = [[0 for _ in range(len(self.competitive_units))] for _ in range(len(self.competitive_units))]
        self.reconstruction_losses = [0] * len(self.competitive_units)
        self.discrimination_losses = [torch.tensor(0.0, device=self.device, dtype=torch.float32) for _ in range(len(self.competitive_units))]
        # if dynamic_loss_weights:
            # self.feedback_weights = [torch.tensor(1.0, device=self.device, dtype=torch.float32) for _ in range(len(self.competitive_units))]
            # self.discrimination_weights = [torch.tensor(0.0, device=self.device, dtype=torch.float32) for _ in range(len(self.competitive_units))]
            # self.reconstruction_weights = [torch.tensor(1.0, device=self.device, dtype=torch.float32) for _ in range(len(self.competitive_units))]
        # else:
        self.feedback_weights = [torch.tensor(feedback_weight, device=self.device, dtype=torch.float32) for _ in range(len(self.competitive_units))]
        self.discrimination_weights = [torch.tensor(discrimination_weight, device=self.device, dtype=torch.float32) for _ in range(len(self.competitive_units))]
        self.reconstruction_weights = [torch.tensor(reconstruction_weight, device=self.device, dtype=torch.float32) for _ in range(len(self.competitive_units))]

        # self.adversarial_loss_function = torch.nn.HuberLoss()
        # self.reconstruction_loss_function = torch.nn.HuberLoss()
        self.adversarial_loss_function = torch.nn.L1Loss()
        self.reconstruction_loss_function = torch.nn.L1Loss()
        # self.adversarial_loss_function = torch.nn.MSELoss()        
        # self.reconstruction_loss_function = torch.nn.MSELoss()
        self.max_roc_auc = 0.0
        self.max_unit_max = 0.0
        self.max_avg_score = 0.0
        self.max_k_score = 0.0
        self.max_revavg_score = 0.0
        self.max_inv_avg_score = 0.0
        self.max_loss_unit_score = 0.0
        self.avg_sum = 0.0
        self.average_sum = 0.0
        self.max_a_priori_roc_auc = 0.0
        self.store_imgs = []
        self.disc_values = np.array([0] * 10)
        self.gen_values = np.array([0] * 10)
        
        self.learning_units = minimum_learning_units

        self.lambda_w = 4.0
        
        self.reconstructions = []

        # weight of reconstruction error of fake images for discriminator (is updated after every batch)
        self.k_discriminator = 0.0 
        self.k_generator = 0.0 
        self.imgs = None
        self.unit_idx = 0
        self.unit_counts = [0] * len(self.competitive_units)
        self.unit_losses = [0] * len(self.competitive_units)
        for unit in self.competitive_units:
            # self.init_weights(unit, init_type="normal")
            self.init_weights(unit, init_type="kaiming")
        self.hparams.feedback_group_size = min(self.hparams.feedback_group_size, len(self.competitive_units) - 1)
        
        self.log_dict({"metrics/traintime": time.time() - self.train_start_time})
        self.competitive_units = ModuleList(self.competitive_units)
        # print(self.competitive_units[0])
    
    def init_weights(self, net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
                torch.nn.init.constant_(m.bias.data, 0.0)
        net.apply(init_func)
        
    def on_fit_start(self) -> None:
        self.competitive_units = ModuleList([unit.to(self.device) for unit in self.competitive_units])
        # for unit in self.competitive_units:
        # if wandb.run is not None:
        #     wandb.watch(self.competitive_units[0], log="all", log_freq=20)
        
        self.discrimination_losses = [a.to(self.device).to(torch.float32) for a in self.discrimination_losses]
        self.feedback_weights = [a.to(self.device).to(torch.float32) for a in self.feedback_weights]
        self.discrimination_weights = [a.to(self.device).to(torch.float32) for a in self.discrimination_weights]
        self.reconstruction_weights = [a.to(self.device).to(torch.float32) for a in self.reconstruction_weights]
    
    # def get_best_unit_pair(self):
        # max_score = 0.0
        # self.obscured_data = self.obscured_data.to(self.device)
        # for i in range(len(self.competitive_units)):
        #     for j in range(len(self.competitive_units)):
                
        #         generator, discriminator = self.competitive_units[i], self.competitive_units[j]
        #         generator.eval()
        #         discriminator.eval()
        #         scnd_reconstruction = discriminator(generator(self.obscured_data))
        #         prediction = self.compute_residual_scores(self.obscured_data, scnd_reconstruction).detach()
        #         prediction = self.normalize_scores(prediction)
        #         score = self.compute_auc(self.obscured_labels, prediction)
        #         if score > max_score:
        #             max_score = score
        #             self.best_gen = i
        #             self.best_disc = j
        # print("found best gen:", self.best_gen, "best disc:", self.best_disc)
        
    
    def on_train_epoch_end(self) -> None:
        c_lr_schedulers = self.lr_schedulers()
        if c_lr_schedulers is not None:
            if isinstance(c_lr_schedulers, list) and len(c_lr_schedulers) > 0:
                for scheduler in c_lr_schedulers:
                    if scheduler is not None:
                        scheduler.step()
            else:
                c_lr_schedulers.step()
        self.log_dict({"metrics/traintime": time.time() - self.train_start_time})
        # self.get_best_unit_pair()
    
    def on_train_start(self) -> None:
        self.reconstructions = []
        gc.collect()
        for unit in self.competitive_units:
            unit.train()
    
    def training_step(self, batch, batch_idx):
        progress = self.global_step / self.hparams.training_steps
        self.learning_units = int(self.hparams.minimum_learning_units + math.floor((len(self.competitive_units) - self.hparams.minimum_learning_units + 1) * progress))
        self.learning_units = min(self.learning_units, len(self.competitive_units))
        self.log("learning_units", self.learning_units)
        self.example_train_batch = batch
        if isinstance(batch, list):
            self.imgs = batch[0]
        else:
            self.imgs = batch
        c_optimizers = self.optimizers()
        if not isinstance(c_optimizers, list):
            c_optimizers = [c_optimizers]

        c_lr_schedulers = self.lr_schedulers()
        if c_lr_schedulers is not None and not isinstance(c_lr_schedulers, list):
            c_lr_schedulers = [c_lr_schedulers]
        
        if self.global_step < self.hparams.warmup_steps:
            for idx, (unit, optimizer) in enumerate(zip(self.competitive_units, c_optimizers)):
                unit.train()
                reconstruction = unit(self.imgs)
                reconstruction_loss = self.reconstruction_loss_function(reconstruction, self.imgs)
                optimizer.zero_grad()
                self.manual_backward(reconstruction_loss)
                optimizer.step()
                self.log_dict({
                    f"reconstruction-loss/{idx}": reconstruction_loss.detach(),
                })
            return
        
        num_units = len(self.competitive_units)
        
        if num_units < 2:
            unit1, unit2 = 0, 0
        else:
            unit1, unit2 = np.random.choice(num_units, 2, replace=False)
        optimizer_1, optimizer_2 = c_optimizers[unit1], c_optimizers[unit2]
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        self.unit_losses[unit1], self.unit_losses[unit2] = 0.0, 0.0
        # generator_idx, discriminator_idx = (0, 1)
        for generator_idx, discriminator_idx in [(unit1, unit2), (unit2, unit1)]:
        # for generator_idx, discriminator_idx in [(unit1, unit2)]:
            self.unit_counts[generator_idx] += 1
            self.unit_counts[discriminator_idx] += 1
            # print("gen idx:", generator_idx, "dic idx:", discriminator_idx)
            
            # generator_idx = self.unit_idx
            # discriminator_idx = (generator_idx + 1) % num_units
            generator, discriminator = self.competitive_units[generator_idx], self.competitive_units[discriminator_idx]
            generator.train()
            discriminator.train()
        
            generator_reconstruction = generator(self.imgs)
            discriminator_feedback = discriminator(generator_reconstruction)
            
            generator_reconstruction_loss = self.reconstruction_loss_function(self.imgs, generator_reconstruction)
            generator_adversarial_loss = self.adversarial_loss_function(generator_reconstruction, discriminator_feedback.detach())
            # generator_loss = self.lambda_w + self.feedback_weights[generator_idx] * generator_adversarial_loss
            # generator_loss = self.feedback_weights[generator_idx] * generator_adversarial_loss
            generator_loss = self.reconstruction_weights[generator_idx] * generator_reconstruction_loss + self.feedback_weights[generator_idx] * generator_adversarial_loss
            self.unit_losses[generator_idx] += generator_loss.item()
            # generator_loss.backward()
            self.manual_backward(generator_loss)

            discriminator_reconstruction = discriminator(self.imgs)
            discriminator_fake_reconstruction = discriminator(generator_reconstruction.detach())
            discriminator_reconstruction_loss = self.reconstruction_loss_function(self.imgs, discriminator_reconstruction)
            discriminator_adversarial_loss = self.adversarial_loss_function(generator_reconstruction.detach(), discriminator_fake_reconstruction)
            discriminator_loss = self.lambda_w - self.discrimination_weights[discriminator_idx] * discriminator_adversarial_loss
            self.unit_losses[discriminator_idx] += (discriminator_loss.item() - self.lambda_w)
            # discriminator_loss = self.reconstruction_weights[discriminator_idx] * discriminator_reconstruction_loss - (self.discrimination_weights[discriminator_idx] * discriminator_adversarial_loss)
            # discriminator_loss = self.lambda_w + discriminator_reconstruction_loss - (self.discrimination_weights[discriminator_idx] * discriminator_adversarial_loss)
            
            # discriminator_loss.backward()
            self.manual_backward(discriminator_loss)
            
            diff = self.hparams.gamma_crn * discriminator_reconstruction_loss.detach() - discriminator_adversarial_loss.detach()
            if self.hparams.dynamic_loss_weights:
                self.discrimination_weights[discriminator_idx] += self.hparams.k_lambda * diff
                self.discrimination_weights[discriminator_idx] = min(1.0, max(0.0, self.discrimination_weights[discriminator_idx]))
                

            self.store_imgs.append((self.imgs, generator_reconstruction.detach()))
            if len(self.store_imgs) > 3:
                self.store_imgs = self.store_imgs[1:]
            if batch_idx % self.hparams.image_output_interval == 0:
                self.store_images(batch_idx)
            
            convergence = discriminator_reconstruction_loss.detach() + abs(diff)
            self.convergences[discriminator_idx] = convergence.item()
            self.generator_losses[generator_idx] = generator_loss.item()
            self.reconstruction_losses[generator_idx] = generator_reconstruction_loss.item()
            self.discriminator_losses[discriminator_idx] = discriminator_adversarial_loss.item() - discriminator_reconstruction_loss.item()
            self.feedback_losses[generator_idx][discriminator_idx] = generator_adversarial_loss.item()
            self.recon_feedback_diff[generator_idx][discriminator_idx] = generator_adversarial_loss.item() - generator_reconstruction_loss.item()
            
            self.log_dict({
                f"convergence/{discriminator_idx}": convergence,
                # f"feedback-weights/{generator_idx}": self.feedback_weights[generator_idx],
                f"discrimination-weights/{generator_idx}": self.discrimination_weights[generator_idx],
                # f"reconstruction-weights/{generator_idx}": self.reconstruction_weights[generator_idx],
                f"reconstruction-loss/{generator_idx}": generator_reconstruction_loss.detach(),
                f"feedback-loss/{generator_idx}": generator_adversarial_loss.detach(),
                f"diffs/{generator_idx}-{discriminator_idx}": self.recon_feedback_diff[generator_idx][discriminator_idx],

                # f"lr_{generator_idx}": c_lr_schedulers[generator_idx].get_last_lr()[0] if c_lr_schedulers is not None else self.hparams.lr,
                f"losses/gen_{generator_idx}": generator_loss.detach(),
                f"losses/disc_{discriminator_idx}": discriminator_loss.detach(),
            })
        optimizer_1.step()
        optimizer_2.step()
        self.log_dict({
            f"unit_losses/loss_{unit1}": self.unit_losses[unit1],
            f"unit_losses/loss_{unit2}": self.unit_losses[unit2],
        })
        
        
            
        gc.collect()


    def on_validation_epoch_start(self) -> None:
        self.preds = torch.Tensor([]).to(self.device)
        self.targets = torch.Tensor([]).to(self.device)

        self.best_gen = np.argmin(self.reconstruction_losses)
        self.best_disc = np.argmax(self.discriminator_losses)
        self.log_dict({
            "best_gen": self.best_gen,
            "best_disc": self.best_disc,
        })
        for unit in self.competitive_units:
            unit.eval()
    
    def compute_residual_scores(self, image, reconstruction):
        return torch.mean(torch.abs(image - reconstruction), dim=[1, 2, 3])
    
    def get_difference_image(self, image, reconstructed):
        difference = torch.abs(image - reconstructed) - 1.0
        return torch.mean(difference, dim=1).unsqueeze(1).repeat(1, 3, 1, 1)

    def validation_step(self, batch, batch_idx):
        
        if self.global_step < self.hparams.warmup_steps:
            return
        image, target = batch
 
        target = target.to(int)
        # self.targets += list(target)
        self.targets = torch.cat((self.targets, target))

        generator, discriminator = self.competitive_units[self.best_gen], self.competitive_units[self.best_disc]
        generator.eval()
        discriminator.eval()
        
        reconstruction = generator(image).detach()
        discrimination = discriminator(reconstruction).detach()
        residual_score = self.compute_residual_scores(reconstruction, discrimination)
        self.preds = torch.cat((self.preds, residual_score))
        self.opreds = torch.cat((self.opreds, 2.0 - self.compute_residual_scores(image, discrimination)))
        if batch_idx % self.hparams.image_output_interval == 0:
            # print("now here")
            self.store_images_from_input(
                image, reconstruction,
                self.get_difference(image, reconstruction),
                discrimination,
                self.get_difference(reconstruction, discrimination),
                batch_index=batch_idx,
                tag=",".join([str(a.item()) for a in target][:5]) + "--" + ",".join([f"{a.item():.2f}" for a in residual_score][:5])
            )
        # self.log_dict(log, prog_bar=False)
    
    def compute_auc(self, target, prediction):
        try:
            return roc_auc_score(target.cpu().numpy(), prediction.cpu().numpy())
        except Exception as e:
            print("Exception computing roc: ", e)
            return 0.0
    
    def normalize_scores(self, scores):
        min_score, max_score = torch.min(scores), torch.max(scores)
        return (scores - min_score) / (max_score - min_score)

    def rank_normalize(self, scores):
        _, ranks = torch.sort(scores)
        # print("scores before:", scores)
        for i in range(len(scores)):
            scores[ranks[i]] = i
        # print("scores after:", scores)
        return scores / (len(scores) - 1)

    
    def validation_epoch_end(self, outputs) -> None:        
        if self.global_step < self.hparams.warmup_steps:
            
            self.log_dict({
                "metrics/max_roc_auc": 0.0,
            }, prog_bar=True)
            return
        score = self.compute_auc(self.targets, self.rank_normalize(self.preds))
        revascore = max(score, 1.0 - score)
        oscore = self.compute_auc(self.targets, self.rank_normalize(self.opreds))
        
        print("max unit score:", self.max_a_priori_roc_auc, "avg score:", revascore)
        inv_score = 1.0 - score
        self.max_roc_auc = max(self.max_roc_auc, inv_score)
        self.max_inv_avg_score = max(self.max_inv_avg_score, inv_score)
        self.max_a_priori_roc_auc = max(self.max_a_priori_roc_auc, revascore)
        self.log_dict({
            "metrics/max_roc_auc": self.max_roc_auc,
        }, prog_bar=True)
        self.max_avg_score = max(self.max_avg_score, score)
        self.max_revavg_score = max(self.max_revavg_score, revascore)
        self.log_dict({
            "metrics/max_avg_score": self.max_avg_score,
            "metrics/max_revavg_score": self.max_revavg_score,
            "metrics/max_inv_avg_score": self.max_inv_avg_score,
            "metrics/max_k_score": self.max_k_score,
            "metrics/max_o_score": self.max_roc_auc,
            "metrics/score": score,
            "metrics/invscore": inv_score,
            "metrics/oscore": oscore,
            "metrics/revascore": revascore,
            "metrics/max_a_priori": self.max_a_priori_roc_auc,
            
            "metrics/traintime": time.time() - self.train_start_time,
        })
        gc.collect()

        self.preds = torch.Tensor([]).to(self.device)
        self.targets = torch.Tensor([]).to(self.device)

        self.store_images()
    
    def simple_normalize(self, tensor):
        # return tensor
        return (tensor + 1.0) / 2.0

    def store_images(self, batch_index=None):
        if self.image_output_path is None:
            return
        if self.imgs is None:
            return
        if len(self.store_imgs) == 0:
            return
        if batch_index is None:
            batch_index = "full-epoch"
        max_num_images = 5
        images = []
        for store in self.store_imgs:
            img, reconstruction = store
            difference = self.get_difference(img[0].unsqueeze(0), reconstruction[0].unsqueeze(0))
            images.append(self.simple_normalize(img[0]))
            images.append(self.simple_normalize(reconstruction[0]))
            images.append(self.simple_normalize(difference.squeeze()))
        
        grid = make_grid(images, nrow=3 , normalize=False, padding=20, pad_value=1.0)
        if isinstance(batch_index, int):
            batch_index = f"{batch_index:04d}"
        save_image(grid, self.image_output_path / f"{self.current_epoch:03d}-{batch_index}.png", normalize=False)

        
    def store_images_from_input(self, imgs, *reconstructions, batch_index=None, tag=""):
        if self.image_output_path is None:
            return
        if batch_index is None:
            batch_index = "from-input-full-epoch"
        max_num_images = 5
        num_images = min(max_num_images, len(imgs))
        images = []
        for i in range(num_images):
            images.append(self.simple_normalize(imgs[i]))
            # for j in range(len(reconstructions)):
            #     images.append(self.simple_normalize(reconstructions[j][i]))
        grid = make_grid(images, nrow=len(reconstructions) + 1 , normalize=False, padding=10, pad_value=0.0)
        if isinstance(batch_index, int):
            batch_index = f"from-input-{batch_index:04d}-{tag}"
        save_image(grid, self.image_output_path / f"{self.current_epoch:03d}-{batch_index}.png", normalize=False)
    
    
    def inference(self, inference_data, device, shall_clean=False):
        scores = []
        logging.info(f"starting inference of {len(inference_data)} images...")
        inference_path = self.image_output_path / "inference"
        inference_reconstructed = self.image_output_path / "reconstructed"
        inference_diff = self.image_output_path / "diff"
        if shall_clean:
            logging.info("cleaning target directories....")
        for inference_picture_dir in [inference_path, inference_reconstructed, inference_diff]:
            inference_picture_dir.mkdir(parents=True, exist_ok=True)
            if shall_clean:
                for inference_picture in inference_picture_dir.iterdir():
                    inference_picture.unlink()
        # self.generator.eval()
        # unit = self.competitive_units[argmin(self.convergences)]
        # unit = unit.to(device)
        self.competitive_units = [unit.to(device) for unit in self.competitive_units]
        logging.info(f"inference pictures will be outputed to {str(inference_path)}")
        for image, image_name in tqdm(inference_data, total=len(inference_data)):
            image = image.to(device)
            
            reconstructed = None
            for unit in self.competitive_units:
                if reconstructed is None:
                    reconstructed = unit(image)
                else:
                    reconstructed = torch.maximum(reconstructed, unit(image))
            # reconstructed /= num_units
            residual_score = self.compute_residual_scores(image, reconstructed).item()
            difference = self.get_difference(image, reconstructed)
            # print("image:", torch.min(image).item(), torch.max(image).item())
            # print("reconstruction:", torch.min(reconstructed).item(), torch.max(reconstructed).item())
            # print("diff:", torch.min(difference).item(), torch.max(difference).item())
            # print(image.shape, reconstructed.shape, difference.shape)

            grid = make_grid(
                [
                    self.simple_normalize(image.squeeze()),
                    self.simple_normalize(reconstructed.squeeze()),
                    self.simple_normalize(difference.squeeze()),
                ], 
                nrow=3, normalize=False, padding=2, pad_value=1)
            save_image(grid, inference_path / f"{image_name[0]}.png")
            
            # print(inference_data.dataset.orig_image_size)
            rescale = transforms.Resize(inference_data.dataset.orig_image_size[:2])
            difference = rescale(difference.squeeze())
            save_image(self.simple_normalize(difference[0].unsqueeze(0)), inference_diff / f"{image_name[0]}.png", normalize=False)
            
            scores.append(residual_score)
        return scores

    def configure_optimizers(self):
        print("using optimizer:", self.hparams.optimizer)
        print("using lr scheduler:", self.hparams.lr_scheduler)
        c_optimizers = [create_optimizer(self.hparams.optimizer, self.competitive_units[i], self.hparams.lr, self.hparams.momentum) for i in range(self.hparams.num_competitive_units)]
        if self.hparams.lr_scheduler is None or self.hparams.lr_scheduler == "":
            return c_optimizers, []
        print("################ setting epochs for lr schedulers:", self.hparams.epochs)
        c_lr_schedulers = [create_scheduler(self.hparams.lr_scheduler, c_optimizers[i], self.hparams.lr_factor, self.hparams.lr_steps, self.hparams.epochs, int(self.hparams.training_steps / self.hparams.num_competitive_units)) for i in range(self.hparams.num_competitive_units)]
        return c_optimizers, c_lr_schedulers
