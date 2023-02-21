import wandb
from sklearn.metrics import roc_auc_score
from pathlib import Path
import logging
from tqdm import tqdm
from typing import List

import torch
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid, save_image
import math

from training.networks import RoadImageDiscriminator, create_optimizer, create_scheduler
from training.skipnetworks import UnetGenerator, get_norm_layer

USE_SKIP = True

class DAGAN(LightningModule):
    def __init__(
            self,
            input_shape,
            image_output_interval: int,
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
            **kwargs):
        super(DAGAN, self).__init__()
        self.save_hyperparameters()
        
        self.automatic_optimization = False
        if image_output_path is not None:
            self.image_output_path = Path(image_output_path)
            self.image_output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.image_output_path = None
        
        network_depth = int(math.log2(input_shape[2]))
        self.generator = UnetGenerator(input_shape[1], input_shape[1], network_depth, norm_layer=get_norm_layer(norm), use_dropout=use_dropout)
        network_depth_disc = 2
        self.discriminator = RoadImageDiscriminator(input_shape, num_levels=network_depth_disc)

        self.adversarial_loss_function = torch.nn.MSELoss()
        self.contextual_loss_function = torch.nn.MSELoss()
        self.latent_loss_function = torch.nn.L1Loss()
        
        self.residual_loss_function = torch.nn.L1Loss()
        self.evaluation_loss_function = torch.nn.BCELoss()
        
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
    
        self.reconstructed = self(self.imgs)

        self.fake_discrimination, _ = self.discriminator(self.reconstructed)

        generator_adversarial_loss = self.adversarial_loss_function(self.reconstructed, self.fake_discrimination.detach())
        generator_reconstruction_loss = self.contextual_loss_function(self.imgs, self.reconstructed)

        generator_loss = (
            self.hparams.adversarial_loss_weight * generator_adversarial_loss +
            self.hparams.contextual_loss_weight * generator_reconstruction_loss
        )
        self.manual_backward(generator_loss)
        generator_optimizer.step()

        self.fake_discrimination, _ = self.discriminator(self.reconstructed.detach())
        self.real_discrimination, _ = self.discriminator(self.imgs)

        discriminator_reconstruction_loss = self.adversarial_loss_function(self.real_discrimination, self.imgs)
        discriminator_adversarial_loss = self.adversarial_loss_function(self.fake_discrimination, self.reconstructed.detach())

        discriminator_loss = self.hparams.discriminator_reconstruction_loss_weight * discriminator_reconstruction_loss - self.hparams.adversarial_loss_weight * discriminator_adversarial_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        if batch_idx % self.hparams.image_output_interval == 0:
            self.store_images(batch_idx)

        if len(self.reconstructed) > 1:
            # some debugging stats
            self.log_dict({
                "std/img int": (self.imgs[:].std()),
                "std/reconstructed high": (self.reconstructed[:].std()),
                "std/discrimination high": (self.fake_discrimination[:].std()),
                "gadvloss": generator_adversarial_loss.detach(),
                "gconloss": generator_reconstruction_loss.detach(),
                "darl": discriminator_reconstruction_loss.detach(),
                "dafl": discriminator_adversarial_loss.detach(),
            })
        self.log_dict({
            "loss": (generator_loss.detach() + discriminator_loss.detach()),
            "gloss": generator_loss.detach(),
            "dloss": discriminator_loss.detach(),
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
        self.preds = []
        self.targets = []
    
    def validation_step(self, batch, batch_idx):
        image, target = batch
 
        target = target.to(int)
        reconstruction = self.generator(image)
        discrimination, _ = self.discriminator(image)
        residual_score = (
            torch.mean(torch.abs(image - reconstruction), dim=[1, 2, 3]) +
            torch.mean(torch.abs(image - discrimination), dim=[1, 2, 3])
        ) / 2.0
        self.preds += list(residual_score.detach().cpu())

        target = target.cpu()
        self.targets += list(target)

        if batch_idx % self.hparams.image_output_interval == 0:
            self.store_images_from_input(batch_idx, image, reconstruction, discrimination, ",".join([str(a.item()) for a in list(target)]))
    
    def validation_epoch_end(self, outputs) -> None:

        min_pred, max_pred = min(predictions), max(predictions)
        predictions = [(pred - min_pred) / (max_pred - min_pred) for pred in predictions]

        try:
            score = roc_auc_score(self.targets, predictions)
        except:
            score = 0.0

        self.max_auc_roc = max(self.max_auc_roc, score)
        self.log_dict({
            f"metrics/roc_auc_function": score,
            f"metrics/test_loss": self.evaluation_loss_function(torch.Tensor(self.targets), torch.Tensor(predictions))
        })
        wandb.log({
            f"debug/prediction_scores": wandb.Histogram(predictions),
        })
        
        self.log_dict({
            "metrics/max_roc_auc": self.max_auc_roc,
        }, prog_bar=True)
        self.preds = []
        self.targets = []
        
    
    def simple_normalize(self, tensor):
        return (tensor + 1.0) / 2.0

    def get_difference(self, image, reconstructed, discrimination):
        difference = torch.clamp(
            (torch.abs(image - reconstructed) + torch.abs(image - discrimination)) / 2.0, 
        0.0, 1.0)
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
            reconstructed = self.generator(image)
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


