import gc
import wandb
from sklearn.metrics import roc_auc_score
from numpy import argmin
from pathlib import Path
import logging
from tqdm import tqdm
from typing import List, Optional, Union

import torch
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid, save_image
import math

from training.networks import ORIGINAL, RoadImageUNet, RoadImageDiscriminator, DOUBLE_SKIP, BALANCED_GENERATOR, ONLY_DISC, create_optimizer, create_scheduler
from training.skipnetworks import UnetGenerator, get_norm_layer, UnetGeneratorImproved

USE_SKIP = True

class RoadAnomalyDetector(LightningModule):
    def __init__(
            self,
            input_shape,
            gamma_disc: float,
            gamma_gen: float,
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
            network_depth: int,
            use_dropout: bool,
            norm: str,
            generator_network_depth: int,
            stride: int,
            conv_bias: bool,
            arch: ORIGINAL,
            training_mode=ORIGINAL,
            validation_function=0,
            **kwargs):
        super(RoadAnomalyDetector, self).__init__()
        self.save_hyperparameters()
        
        print("GOT CONV BIAS:", conv_bias)
        self.automatic_optimization = False
        if image_output_path is not None:
            self.image_output_path = Path(image_output_path)
            self.image_output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.image_output_path = None
        
        if USE_SKIP:
            network_depth = int(math.log2(input_shape[2]))
            self.generator = UnetGenerator(input_shape[1], input_shape[1], network_depth, norm_layer=get_norm_layer(norm), use_dropout=use_dropout)
        else:
            self.generator = RoadImageUNet(input_shape, levels=generator_network_depth, stride=stride, arch=arch)
        if arch == ORIGINAL:
            network_depth_disc = 2
            # network_depth_disc = int(math.log2(input_shape[2]))
            self.discriminator = RoadImageDiscriminator(input_shape, num_levels=network_depth_disc)
            print(self.discriminator)
        else:
            if USE_SKIP:
                network_depth_disc = int(network_depth / 2)
                self.discriminator = UnetGenerator(input_shape[1], input_shape[1], network_depth_disc, norm_layer=get_norm_layer(norm), use_dropout=use_dropout)
            else:
                network_depth_disc = int(math.log2(input_shape[2]))
                # network_depth_disc = int(network_depth / 2)
                self.discriminator = RoadImageUNet(input_shape, levels=network_depth_disc, stride=stride, arch=arch)

        if arch == ORIGINAL:
            self.adversarial_loss_function = torch.nn.MSELoss()
            self.contextual_loss_function = torch.nn.MSELoss()
        else:
            self.adversarial_loss_function = torch.nn.L1Loss()
            self.contextual_loss_function = torch.nn.L1Loss()
        self.latent_loss_function = torch.nn.L1Loss()
        
        self.residual_loss_function = torch.nn.L1Loss()
        self.evaluation_loss_function = torch.nn.BCELoss()
        
        
        # self.f1 = F1Score(num_classes=2)
        # self.prec = Precision(num_classes=2)
        # self.recall = Recall(num_classes=2)
        # self.auc_roc = AUROC()
        self.max_auc_roc = 0.0

        # weight of reconstruction error of fake images for discriminator (is updated after every batch)
        self.k_discriminator = 0.0 
        self.k_generator = 0.0 
        self.imgs = None
        
        # self.init_weights()
    
    def init_weights(self) -> None:
        for module in self.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                if module.kernel_size[0] == 7:
                    # first conv layer
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    # other 32-bit conv layers
                    torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
    
    
    def forward(self, input):
        return self.generator(input)
    
    
    def training_step(self, batch, batch_idx):
        self.imgs = batch
        generator_optimizer, discriminator_optimizer = self.optimizers()
        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        if USE_SKIP:
            self.reconstructed = self(self.imgs)
        else:
            self.reconstructed, _ = self(self.imgs)

        if USE_SKIP and self.hparams.arch != ORIGINAL:
            self.fake_discrimination = self.discriminator(self.reconstructed)
        else:
            self.fake_discrimination, _ = self.discriminator(self.reconstructed)

        generator_adversarial_loss = self.adversarial_loss_function(self.reconstructed, self.fake_discrimination.detach())
        generator_reconstruction_loss = self.contextual_loss_function(self.imgs, self.reconstructed)

        if self.hparams.training_mode == BALANCED_GENERATOR:
            generator_loss = generator_reconstruction_loss + self.k_generator * generator_adversarial_loss
        else:
            generator_loss = (
                self.hparams.adversarial_loss_weight * generator_adversarial_loss +
                self.hparams.contextual_loss_weight * generator_reconstruction_loss
            )
        self.manual_backward(generator_loss)
        generator_optimizer.step()

        if USE_SKIP and self.hparams.arch != ORIGINAL:
            self.fake_discrimination = self.discriminator(self.reconstructed.detach())
            self.real_discrimination = self.discriminator(self.imgs)
        else:
            self.fake_discrimination, _ = self.discriminator(self.reconstructed.detach())
            self.real_discrimination, _ = self.discriminator(self.imgs)

        discriminator_reconstruction_loss = self.adversarial_loss_function(self.real_discrimination, self.imgs)
        discriminator_adversarial_loss = self.adversarial_loss_function(self.fake_discrimination, self.reconstructed.detach())

        if self.hparams.training_mode == ORIGINAL:
            discriminator_loss = self.hparams.discriminator_reconstruction_loss_weight * discriminator_reconstruction_loss - self.hparams.adversarial_loss_weight * discriminator_adversarial_loss
        else:
            discriminator_loss = (discriminator_reconstruction_loss - self.k_discriminator * discriminator_adversarial_loss)
        discriminator_loss.backward()
        discriminator_optimizer.step()


        diff_discriminator = self.hparams.gamma_disc * discriminator_reconstruction_loss - discriminator_adversarial_loss
        # Update weight term for fake samples
        self.k_discriminator = self.k_discriminator + self.hparams.k_lambda * diff_discriminator.item()
        self.k_discriminator = min(max(self.k_discriminator, 0), 1)  # Constraint to interval [0, 1]

        diff_generator = self.hparams.gamma_gen * generator_reconstruction_loss - generator_adversarial_loss
        # Update weight term for fake samples
        self.k_generator = self.k_generator + self.hparams.k_lambda * diff_generator.item()
        self.k_generator = min(max(self.k_generator, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        convergence = (
            discriminator_reconstruction_loss +
            generator_reconstruction_loss +
            torch.abs(diff_generator) + 
            torch.abs(diff_discriminator)).item()

        if batch_idx % self.hparams.image_output_interval == 0:
            self.store_images(batch_idx)

        if len(self.reconstructed) > 1:
            # some debugging stats
            self.log_dict({
                "std/img int": (self.imgs[:].std()),
                "diff_generator": diff_generator,
                "diff_discriminator": diff_discriminator,
                "std/reconstructed high": (self.reconstructed[:].std()),
                "std/discrimination high": (self.fake_discrimination[:].std()),
                "gadvloss": generator_adversarial_loss.detach(),
                "gconloss": generator_reconstruction_loss.detach(),
                "darl": discriminator_reconstruction_loss.detach(),
                "dafl": discriminator_adversarial_loss.detach(),
                "k_generator": self.k_generator,
                "k_discriminator": self.k_discriminator,
            })
        self.log_dict({
            "loss": (generator_loss.detach() + discriminator_loss.detach()),
            "gloss": generator_loss.detach(),
            "dloss": discriminator_loss.detach(),
            "convergence": convergence, 
        }, prog_bar=False)
    
    def on_epoch_end(self) -> None:
        self.store_images()
        lr_schedulers = self.lr_schedulers()
        # if self.generator_lr_scheduler is not None:
        #     self.generator_lr_scheduler.step()
        #     self.discriminator_lr_scheduler.step()
        if lr_schedulers is not None:
            for scheduler in lr_schedulers:
                if scheduler is not None:
                    scheduler.step()
    
    def on_validation_epoch_start(self) -> None:
        self.preds = [[] for _ in range(1)]
        self.targets = []
    
    def validation_step(self, batch, batch_idx):
        image, target = batch
 
        target = target.to(int)
        if USE_SKIP:
            reconstruction = self.generator(image)
            if self.hparams.arch != ORIGINAL:
                discrimination = self.discriminator(image)
            else:
                discrimination, _ = self.discriminator(image)
        else:
            reconstruction, _ = self.generator(image)
            discrimination, _ = self.discriminator(image)
        # residual_score = torch.mean(self.residual_loss_function(image, reconstruction), dim=[1, 2, 3])
    # if self.hparams.validation_function == ORIGINAL:
        # residual_score = torch.mean(torch.pow(image - reconstruction, 2), dim=[1, 2, 3])
        # self.preds[0] += list(residual_score.detach().cpu())
        # residual_score = self.residual_loss_function(image, reconstruction)
        # residual_score = self.residual_loss_function(image, reconstruction)
    # elif self.hparams.validation_function == DOUBLE_SKIP:
        residual_score = (
            torch.mean(torch.abs(image - reconstruction), dim=[1, 2, 3]) +
            torch.mean(torch.abs(image - discrimination), dim=[1, 2, 3])
        ) / 2.0
        self.preds[0] += list(residual_score.detach().cpu())
    # elif self.hparams.validation_function == BALANCED_GENERATOR:
    #     residual_score = (
    #         torch.mean(torch.pow(image - reconstruction, 2), dim=[1, 2, 3]) +
    #         torch.mean(torch.pow(image - discrimination, 2), dim=[1, 2, 3])
    #     ) / 2.0
    #     self.preds[2] += list(residual_score.detach().cpu())
    # # elif self.hparams.validation_function == ONLY_DISC:
    #     residual_score = torch.mean(torch.pow(image - discrimination, 2), dim=[1, 2, 3])
    #     self.preds[3] += list(residual_score.detach().cpu())
            # residual_score = (
            #     self.contextual_loss_function(image, reconstruction) +
            #     self.contextual_loss_function(image, discrimination)
            # ) / 2.0
            # residual_score = (
            #     torch.mean(torch.pow(torch.abs(image - reconstruction), 2), dim=[1, 2, 3]) +
            #     torch.mean(torch.pow(torch.abs(image - discrimination), 2), dim=[1, 2, 3])
            # ) / 8.0
        target = target.cpu()
        # residual_score = residual_score.cpu()
        # self.auc_roc.update(residual_score, target)
        # self.preds += list(residual_score)
        self.targets += list(target)
        # print(target, residual_score)
        if batch_idx % self.hparams.image_output_interval == 0:
            self.store_images_from_input(batch_idx, image, reconstruction, discrimination, ",".join([str(a.item()) for a in list(target)]))
        # print("target:", target, "residual", residual_score)
        # try:
        #     roc_auc = roc_auc_score(target, residual_score)
        # except:
        #     roc_auc = 1.0
        # self.log_dict({
            # "debug/residual score max": residual_score.max(),
            # "debug/residual score min": residual_score.min(),
            # "debug/residual score": torch.mean(residual_score),
            # "metrics/precision": self.prec(residual_score, target),
            # "metrics/recall": self.recall(residual_score, target),
            # "metrics/f1": self.f1(residual_score, target),
        # }, prog_bar=False)
        # difference_discriminator, _, _ = self.get_difference(image, discrimination)
    
    def validation_epoch_end(self, outputs) -> None:
        # print(len(self.auc_roc.preds), self.auc_roc.preds[0].shape, np.array(self.auc_roc.preds).shape)
        # min_pred = np.min(np.array(self.auc_roc.preds))
        # max_pred = np.max(np.array(self.auc_roc.preds))
        
        # min_pred, max_pred = 1e10, 0.0
        # for i in self.auc_roc.preds:
        #     min_pred, max_pred = min([*i, min_pred]), max([*i, max_pred])
        # for i in self.auc_roc.preds:
        #     min_pred, max_pred = min([*i, min_pred]), max([*i, max_pred])
        best_performing = -1
        best_score = -1
        avg_true_score = -1
        avg_false_score = -1
        avg_true_full_score = -1
        avg_false_full_score = -1
        for idx, predictions in enumerate(self.preds):
            min_pred, max_pred = min(predictions), max(predictions)
            predictions = [(pred - min_pred) / (max_pred - min_pred) for pred in predictions]
            try:
                score = roc_auc_score(self.targets, predictions)
            except:
                score = 0.0
            if score > best_score:
                best_performing = idx
                best_score = score
                true_sum = 0.0
                true_sum_full = 0.0
                true_count = 0
                for (t, p, fp) in zip(self.targets, predictions, self.preds[idx]):
                    if t == 0:
                        true_sum += p
                        true_sum_full += fp
                        true_count += 1
                avg_true_score = true_sum / float(true_count)
                avg_true_full_score = true_sum_full / float(true_count)
                avg_false_score = float(sum(predictions) - true_sum) / float(len(predictions) - true_count)
                avg_false_full_score = float(sum(self.preds[idx]) - true_sum_full) / float(len(predictions) - true_count)
            self.max_auc_roc = max(self.max_auc_roc, score)
            self.log_dict({
                f"metrics/roc_auc_function_{idx}": score,
                f"metrics/test_loss_{idx}": self.evaluation_loss_function(torch.Tensor(self.targets), torch.Tensor(predictions))
            })
            wandb.log({
                f"debug/prediction_scores_{idx}": wandb.Histogram(predictions),
            })
            wandb.log({
                f"debug/prediction_full_scores_{idx}": wandb.Histogram(self.preds[idx]),
            })
        
        self.log_dict({
            "metrics/best_function": best_performing,
            "debug/avg_true_score": avg_true_score,
            "debug/avg_false_score": avg_false_score,
            "debug/avg_diffs": avg_false_score - avg_true_score,
            "debug/avg_full_true_score": avg_true_full_score,
            "debug/avg_full_false_score": avg_false_full_score,
            "debug/avg_full_diffs": avg_false_full_score - avg_true_full_score,
        })
        self.log_dict({
            "metrics/roc_auc": best_score,
            "metrics/max_roc_auc": self.max_auc_roc,
        }, prog_bar=True)
        # self.log_dict({
        #     "debug/min pred": min_pred,
        #     "debug/max pred": max_pred,
        #     "debug/test test": dscore
        # }, prog_bar=False)
        # wandb.log({
            # "metrics/roc_plot": wandb.plot.roc_curve(self.targets, self.preds),
            # "metrics/precision_recall": wandb.plot.pr_curve(self.targets, self.preds),
            # "metrics/confusion_matrix": wandb.plot.confusion_matrix(y_true=self.targets, preds=self.preds),
        # })
        self.preds = [[] for _ in range(1)]
        self.targets = []
        
    
    def simple_normalize(self, tensor):
        return (tensor + 1.0) / 2.0

    def get_difference(self, image, reconstructed, discrimination):
        difference = torch.clamp(
            (torch.abs(image - reconstructed) + torch.abs(image - discrimination)) / 2.0, 
        0.0, 1.0)
        # difference = torch.mean(difference, dim=1)
        return difference

    def store_images(self, batch_index=None):
        if self.image_output_path is None:
            return
        if self.imgs is None:
            return
        if batch_index is None:
            batch_index = "full-epoch"
        max_num_images = 5
        num_images = min(max_num_images, len(self.imgs))
        images = []
        for i in range(num_images):
            difference = self.get_difference(self.imgs[i].unsqueeze(0), self.reconstructed[i].unsqueeze(0), self.fake_discrimination[i].unsqueeze(0))
            images.append(self.simple_normalize(self.imgs[i]))
            images.append(self.simple_normalize(self.reconstructed[i]))
            images.append(self.simple_normalize(self.real_discrimination[i]))
            images.append(self.simple_normalize(self.fake_discrimination[i]))
            images.append(difference.squeeze())
        
        grid = make_grid(images, nrow=5 , normalize=False, padding=20, pad_value=1.0)
        if isinstance(batch_index, int):
            batch_index = f"{batch_index:04d}"
        save_image(grid, self.image_output_path / f"{self.current_epoch:03d}-{batch_index}.png", normalize=False)

        
    def store_images_from_input(self, batch_index=None, imgs=None, reconstructed=None, discrimination=None, tag=""):

        if self.image_output_path is None:
            return
        if batch_index is None:
            batch_index = "from-input-full-epoch"
        max_num_images = 5
        num_images = min(max_num_images, len(imgs))
        images = []
        for i in range(num_images):
            difference = self.get_difference(imgs[i].unsqueeze(0), reconstructed[i].unsqueeze(0), discrimination[i].unsqueeze(0))
            images.append(self.simple_normalize(imgs[i]))
            images.append(self.simple_normalize(reconstructed[i]))
            images.append(self.simple_normalize(discrimination[i]))
            images.append(difference.squeeze())
        
        grid = make_grid(images, nrow=4 , normalize=False, padding=20, pad_value=1.0)
        if isinstance(batch_index, int):
            batch_index = f"from-input-{batch_index:04d}"
            # batch_index = f"from-input-{batch_index:04d}-{tag}"
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
        self.generator.eval()
        logging.info(f"inference pictures will be outputed to {str(inference_path)}")
        for image, image_name in tqdm(inference_data, total=len(inference_data)):
            # print(image_name)
            image = image.to(device)
            if USE_SKIP:
                reconstructed = self.generator(image)
            else:
                reconstructed, _ = self.generator(image)
            residual_score = self.contextual_loss_function(image, reconstructed).item()
            difference = self.get_difference(image, reconstructed)

            grid = make_grid([image, reconstructed], nrow=3, normalize=False, padding=2, pad_value=1)
            save_image(grid, inference_path / f"{image_name[0]}.png")
            save_image(difference[0].unsqueeze(0), inference_diff / f"{image_name[0]}.png", normalize=False)
            
            scores.append(residual_score)
        return scores

    def configure_optimizers(self):
        print("using optimizer:", self.hparams.optimizer)
        print("using lr scheduler:", self.hparams.lr_scheduler)
        generator_optimizer = create_optimizer(self.hparams.optimizer, self, self.hparams.lr, self.hparams.momentum)
        discriminator_optimizer = create_optimizer(self.hparams.optimizer, self.discriminator, self.hparams.lr, self.hparams.momentum)
        if self.hparams.lr_scheduler is None or self.hparams.lr_scheduler == "":
            return [generator_optimizer, discriminator_optimizer], []
        generator_lr_scheduler = create_scheduler(self.hparams.lr_scheduler, generator_optimizer, self.hparams.lr_factor, self.hparams.lr_steps, self.hparams.epochs, self.hparams.training_steps)
        discriminator_lr_scheduler = create_scheduler(self.hparams.lr_scheduler, discriminator_optimizer, self.hparams.lr_factor, self.hparams.lr_steps, self.hparams.epochs, self.hparams.training_steps)
        
        return [generator_optimizer, discriminator_optimizer], [generator_lr_scheduler, discriminator_lr_scheduler]


