from typing import Optional, Union

from torch.optim import Adam, SGD, Optimizer, RAdam, AdamW, NAdam, RMSprop
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, MultiStepLR, ExponentialLR
import torch
from torch.nn import (
    Module, Sequential, Conv2d, BatchNorm2d, LeakyReLU, ReLU, ConvTranspose2d, ModuleList, Tanh, Linear, BatchNorm1d, Upsample
)

# ORIGINAL_ARCH = False
# ORIGINAL_TRAIN = False
ORIGINAL = 0
DOUBLE_SKIP = 1
BALANCED_GENERATOR = 2
COMPETITIVE = 3
ONLY_DISC = 4

ARCH = DOUBLE_SKIP
TRAIN = DOUBLE_SKIP

def create_optimizer(name: str, model: Module, lr: float, momentum: float) -> Optimizer:
    """creates the specified optimizer with the given parameters

    Args:
        name (str): str name of optimizer
        model (Module): the model used for training
        lr (float): learning rate
        momentum (float): momentum (only for sgd optimizer)

    Raises:
        ValueError: thrown if optimizer name not known

    Returns:
        Optimizer: the model optimizer
    """
    if name == "adam":
        return Adam(params=model.parameters(), lr=lr)
    if name == "adamw":
        return AdamW(params=model.parameters(), lr=lr)
    if name == "radam":
        return RAdam(params=model.parameters(), lr=lr)
    if name == "nadam":
        return NAdam(params=model.parameters(), lr=lr)
    if name == "rmsprop":
        return RMSprop(params=model.parameters(), lr=lr)
    elif name == "sgd":
        return SGD(params=model.parameters(), lr=lr)
    else:
        raise ValueError(f"No optimizer with name {name} found!")


def create_scheduler(
        scheduler_name: Optional[str],
        optimizer: Optimizer,
        lr_factor: float,
        lr_steps: Optional[list],
        epochs: int, steps: int) -> Union[_LRScheduler, None]:
    """creates a learning rate scheduler with the given parameters

    Args:
        scheduler_name (Optional[str]): str name of scheduler or None, in which case None will be returned
        optimizer (Optimizer): the learning optimizer
        lr_factor (float): the learning rate factor
        lr_steps (Optional[list]): learning rate steps for the scheduler to take (only supported for step scheduler)
        epochs (int): number of scheduler epochs (only supported for cosine scheduler)

    Raises:
        ValueError: thrown if step scheduler was chosen but no steps were passed
        ValueError: thrown if scheduler name not known and not None

    Returns:
        Union[_LRScheduler, None]: either the learning rate scheduler object or None if scheduler_name was None
    """
    if epochs is None:
        epochs = steps
    if scheduler_name == "step":
        if not lr_steps:
            raise ValueError("step scheduler chosen but no lr steps passed!")
        return MultiStepLR(optimizer, lr_steps, lr_factor)
    elif scheduler_name == "exponential":
        return ExponentialLR(optimizer, lr_factor)
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, epochs)
    elif not scheduler_name:
        return None
    else:
        raise ValueError(f"no scheduler with name {scheduler_name} found!")



class DoubleConv(Module):
    """Module placed at the beginning and end of the U-Net architecture. Used to transform input channels 
    to specified channel size."""
    def __init__(self, input_channels, output_channels, output_layer = False, bias=False, arch=ORIGINAL):
        super(DoubleConv, self).__init__()
        if arch == ORIGINAL:
            layers = [Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=bias)]
        elif arch != ORIGINAL:
            layers = [
                Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=bias),
                ReLU(),
                BatchNorm2d(output_channels),
                Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=bias),
            ]
        else:
            layers = []
        if output_layer:
            layers.append(Tanh())
        else:
            layers.append(ReLU())
        self.module = Sequential(*layers)

    
    def forward(self, input, additional_input=None):
        # if additional_input is not None:
        #     return self.module(input + additional_input)
        return self.module(input)

class DownUnit(Module):
    """Down scaling unit of U-Net. Input of form input_channels x w x h is transformed to output_channels x w/2 x h/2"""
    def __init__(self, input_channels, output_channels, stride=2, bias=False, arch=ORIGINAL):
        super(DownUnit, self).__init__()
        self.module = Sequential(
            Conv2d(input_channels, output_channels, kernel_size=4, stride=stride, padding=1, bias=bias),
            LeakyReLU(),
            BatchNorm2d(output_channels),
        )
    
    def forward(self, input):
        return self.module(input)

class UpUnit(Module):
    """Upscaling unit of U-Net. transforms input to target shape and specified output channels."""
    def __init__(self, input_channels, output_channels, target_shape, bias, last_one=False, arch=ORIGINAL):
        super(UpUnit, self).__init__()
        if arch == ORIGINAL:
            if last_one:
                self.module = ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
            else:
                self.module = Sequential(
                    ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
                    ReLU(),
                ) 
        else:
            self.module = Sequential(
                Upsample(size=target_shape[-2:]),
                Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=bias),
                ReLU(),
                BatchNorm2d(output_channels),
            )
    
    def forward(self, input_from_lower_module, input_from_skip):
        # incomming data is concatenated along channel dimension        
        united_input = torch.cat([input_from_lower_module, input_from_skip], dim=1)
        return self.module(united_input)

class RoadImageUNet(Module):
    """Unet used for road image anomaly detection."""
    def __init__(self, input_shape, levels=4, stride=2, conv_bias=False, arch=ORIGINAL):
        super(RoadImageUNet, self).__init__()
        self.input_shape = input_shape
        self.stride = stride
        self.conv_bias = conv_bias
        self.arch = arch
        self._build_unet(levels)
    
    def _build_unet(self, levels):
        # build down sampling units, record which shapes are produced (used for correct upscaling)
        num_channels = 64
        self.downs = [DoubleConv(self.input_shape[1], num_channels, arch=self.arch)]
        self.ups = [DoubleConv(num_channels, self.input_shape[1], output_layer=True,  arch=self.arch)]
        example_input = torch.zeros(self.input_shape)
        target_shapes = []
        for _ in range(levels):
            example_input = self.downs[-1](example_input)
            self.downs.append(DownUnit(num_channels, min(num_channels * 2, 512), self.stride, self.conv_bias))
            target_shapes.append(example_input.shape)
            num_channels = min(num_channels * 2, 512)
        
        # build up sampling units using the correct target shapes
        num_channels = 64
        for i in range(levels):
            self.ups.append(UpUnit(min(num_channels * 4, 1024), self.input_shape[1] if (i == 0 and ARCH == ORIGINAL) else num_channels, target_shapes[i], self.conv_bias, last_one=(i == 0),  arch=self.arch))
            num_channels = min(num_channels * 2, 512)
        self.ups.reverse()


        self.downs = ModuleList(self.downs)
        self.ups = ModuleList(self.ups)
    
    def forward(self, input):
        down_values = []
        for down in self.downs:
            input = down(input)
            down_values.append(input)
        
        # down_values.reverse()
        for idx, up in enumerate(self.ups):
            input = up(input, down_values[len(self.downs) - 1 - idx])
        
        # bottleneck latent values are also returned for further analysis
        return input, down_values[0]


class RoadImageDiscriminator(Module):
    """Original discriminator used in DAGAN paper"""
    num_levels = 3
    num_channels = 64
    bottleneck = 512

    def __init__(self, input_shape, num_levels):
        super(RoadImageDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.num_levels = num_levels
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.latent = None
    
    def _build_encoder(self):
        layers = []
        # step_size = (self.bottleneck - self.input_shape[1]) / self.num_levels
        input_channels = self.input_shape[1]
        num_channels = self.num_channels
        layers = [
            Conv2d(input_channels, num_channels, kernel_size=3, padding=1),
        ]
        for _ in range(self.num_levels):
            layers += [
                Conv2d(num_channels, num_channels * 2, kernel_size=4, stride=2, padding=1),
                LeakyReLU(),
                BatchNorm2d(num_channels * 2),
            ]
            num_channels *= 2
        return Sequential(*layers)

    def _build_decoder(self):
        layers = []

        input_channels = self.input_shape[1]
        num_channels = self.num_channels
        layers = [
            Conv2d(num_channels, input_channels, kernel_size=3, padding=1),
        ]
        for _ in range(self.num_levels):
            layers += [
                ReLU(),
                ConvTranspose2d(num_channels* 2, num_channels, kernel_size=4, stride=2, padding=1),
            ]
            num_channels *= 2
        layers.reverse()
        layers.append(Tanh())
        return Sequential(*layers)
    
    def forward(self, input):
        self.latent = self.encoder(input)
        return self.decoder(self.latent), self.latent
