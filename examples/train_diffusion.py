import logging
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import transforms
from torchvision.datasets import CIFAR10

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
# from accelerate import Accelerator # Not using Accelerator directly for simplicity

import yann
from yann.params import HyperParams, Field # Args not needed here
from yann.train import Trainer

log = logging.getLogger(__name__)
# Configure basic logging
logging.basicConfig(level=logging.INFO)

# --- Diffusion Specific Config Classes (can be defined here or imported) ---
class AutoencoderParams(HyperParams):
    model_name: str = "stabilityai/sd-vae-ft-mse"
    scaling_factor: float = 0.18215
    # Add torch_dtype if needed, e.g., torch_dtype: str = "float16"

class UnetParams(HyperParams):
    in_channels: int = 4
    out_channels: int = 4
    layers_per_block: int = 2
    block_out_channels: tuple = (128, 128, 256, 256, 512, 512)
    down_block_types: tuple = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    up_block_types: tuple = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")

class SchedulerParams(HyperParams):
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"

class OptimizerArgs(HyperParams): # Renamed to avoid clash if using Args[...] elsewhere
    lr: float = 1e-4
    weight_decay: float = 1e-6


# --- Combined Params Inheriting from Trainer.Params ---

# Define DiffusionParams inheriting from Trainer.Params and adding specific fields
class DiffusionParams(yann.train.Trainer.Params):
    # --- Inherited Trainer Params (subset shown for clarity) ---
    # model: torch.nn.Module = None # Will be set to the U-Net instance
    # dataset: torch.utils.data.Dataset = None # Will be set to CIFAR10 instance
    # optimizer: torch.optim.Optimizer = None # Will be set to AdamW instance
    # loss: Callable = None # Set to None as loss is handled in custom step
    # epochs: int = 50
    # batch_size: int = 16
    # num_workers: int = 4
    # device: Optional[str] = None
    # amp: bool = True
    # root: str = "./runs/diffusion"
    # name: Optional[str] = None # Optional: Trainer generates one if None
    # seed: Optional[int] = 42
    # callbacks: list = None # Define below

    # --- Diffusion Specific Params ---
    project_name: str = "cifar10-latent-diffusion-inherits"
    vae_config: AutoencoderParams = AutoencoderParams()
    unet_config: UnetParams = UnetParams()
    scheduler_config: SchedulerParams = SchedulerParams()
    optimizer_config: OptimizerArgs = OptimizerArgs() # Using nested config for optimizer args
    image_size: int = 256 # VAE input size
    dataset_path: str = "./data"
    log_interval: int = 100
    checkpoint_interval: int = 1000 # steps

    # --- Overriding/Setting defaults for inherited fields ---
    epochs: int = 50
    batch_size: int = 16 # Example override
    num_workers: int = 4
    amp: bool = True
    loss: Optional[Callable] = None # Explicitly None for diffusion
    seed: Optional[int] = 42
    root: str = "./runs/diffusion" # Set default run dir


    # --- Components to be initialized and passed ---
    # These are not technically HyperParams but are stored here before Trainer init
    # We will instantiate them in `train()` and pass them to Trainer explicitly
    vae_model: Optional[AutoencoderKL] = Field(default=None, type=AutoencoderKL)
    noise_scheduler: Optional[DDPMScheduler] = Field(default=None, type=DDPMScheduler)



# --- Custom Diffusion Step ---

def diffusion_step(trainer: Trainer, batch):
    """
    Custom training step for latent diffusion model.
    Overrides the default trainer.step behavior.
    Accesses components via trainer.params.
    """
    # Access components via trainer.params now
    vae: AutoencoderKL = trainer.params.vae_model
    unet: UNet2DModel = trainer.model # U-Net is the trainer's main model
    noise_scheduler: DDPMScheduler = trainer.params.noise_scheduler
    params: DiffusionParams = trainer.params

    if vae is None or noise_scheduler is None:
        raise ValueError("VAE model or Noise Scheduler not initialized and passed to Trainer params")

    images = batch[0].to(trainer.device)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * params.vae_config.scaling_factor # Use config from params

    noise = torch.randn_like(latents)
    batch_size = latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (batch_size,),
        device=latents.device
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Use AMP setting from trainer.params
    with torch.autocast(device_type=trainer.device.type, enabled=params.amp):
        noise_pred = unet(noisy_latents, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

    trainer.optimizer.zero_grad(set_to_none=params.none_grad) # Use param

    if trainer.grad_scaler: # grad_scaler comes from trainer state based on params.amp
        trainer.grad_scaler.scale(loss).backward()
        if trainer.clip_grad: # clip_grad instance from trainer init based on params.clip_grad
             trainer.grad_scaler.unscale_(trainer.optimizer)
             trainer.clip_grad(unet.parameters())
        trainer.grad_scaler.step(trainer.optimizer)
        trainer.grad_scaler.update()
    else:
        loss.backward()
        if trainer.clip_grad:
            trainer.clip_grad(unet.parameters())
        trainer.optimizer.step()

    if trainer.lr_scheduler and params.lr_batch_step: # Use param
        trainer._lr_scheduler_step(step=trainer.num_steps, metric=loss.item())

    return noise_pred, loss


# --- Main Training Function ---

def train():
    # 1. Initialize Parameters
    params = DiffusionParams()
    # params = DiffusionParams.from_command() # Optional: Load overrides

    if params.seed is not None:
        yann.seed(params.seed)

    # 2. Load/Initialize Components based on Params
    log.info(f"Loading VAE: {params.vae_config.model_name}")
    vae_model = AutoencoderKL.from_pretrained(params.vae_config.model_name)
    vae_model.requires_grad_(False)
    vae_model.eval() # Ensure eval mode

    log.info("Initializing U-Net model")
    unet_model = UNet2DModel(**params.unet_config.to_dict()) # Pass config as dict

    log.info("Initializing Noise Scheduler")
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=params.scheduler_config.num_train_timesteps,
        beta_schedule=params.scheduler_config.beta_schedule
    )

    # Store initialized components in params object before passing to Trainer
    params.vae_model = vae_model
    params.noise_scheduler = noise_scheduler

    # 3. Prepare Dataset
    log.info("Loading CIFAR-10 dataset")
    preprocess = transforms.Compose(
        [
            transforms.Resize(params.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(params.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = CIFAR10(
        root=params.dataset_path, train=True, download=True, transform=preprocess
    )

    # 4. Prepare Optimizer
    optimizer = AdamW(
        unet_model.parameters(),
        lr=params.optimizer_config.lr,
        weight_decay=params.optimizer_config.weight_decay
    )

    # 5. Define Callbacks
    callbacks = [
        yann.callbacks.Log(interval=params.log_interval),
        yann.callbacks.Checkpoint(
            interval=params.checkpoint_interval, metric='loss', mode='min'
        ),
        # TODO: Add image generation callback
    ]

    # 6. Create Trainer, passing the *specific objects* it expects
    log.info("Initializing Yann Trainer")
    trainer = Trainer(
        params=params,          # Pass the *instance* of DiffusionParams
        model=unet_model,       # Explicitly pass the initialized U-Net
        dataset=dataset,        # Explicitly pass the initialized Dataset
        optimizer=optimizer,    # Explicitly pass the initialized Optimizer
        callbacks=callbacks,    # Pass the callbacks list
        # Trainer will use batch_size, epochs, device, amp, root etc. from params instance
        # 'loss' is already None in DiffusionParams definition
        # 'vae_model' and 'noise_scheduler' are now accessible via trainer.params
        name=f"unet-{params.scheduler_config.beta_schedule}-bs{params.batch_size}" # Example name
    )

    # Move VAE to the correct device (Trainer's .to() method now only moves trainer.model and trainer.loss)
    # We need to move components stored in params manually if they are Modules.
    # Alternatively, modify Trainer.to() to look for specific Module fields in params.
    if params.vae_model:
        params.vae_model.to(trainer.device)


    # 7. Override the step method
    trainer.override('step', diffusion_step)

    # 8. Run Training
    log.info(f"Starting training run: {trainer.name} in {trainer.paths.root}")
    trainer.run()
    log.info("Training finished.")


if __name__ == "__main__":
    train()
