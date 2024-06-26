import torch
import numpy as np
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from pathlib import Path

import time

from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('high')

# Try to import lovely_tensors
try:
    import lovely_tensors as lt
    lt.monkey_patch()
except ModuleNotFoundError:
    print("lovely_tensors not found, skipping monkey patching")
    # But not mandatory, pass if lovely tensor is not available
    pass


# Define a function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base="1.3", config_path="config", config_name="pretrain")
def main(cfg: DictConfig):
    # Seed everything
    seed_everything(cfg.seed)

    # Create logger
    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.experiment_name)

    # Log the config
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Instantiate the model
    image_encoder = instantiate(cfg.vision_model.encoder)

    tokenizer = instantiate(cfg.text_model.tokenizer)

    text_encoder = instantiate(cfg.text_model.encoder)

    # Instantiate the dataset
    datamodule = instantiate(cfg.dataset,
                             image_transform=instantiate(cfg.image_transform),
                             text_tokenizer=tokenizer,
                             mlm=(True if "lm_head" in dir(text_encoder) else False),
                             )

    model = instantiate(cfg.vlp_model,
                        vision_model=image_encoder,
                        text_model=text_encoder,
                        sep_token_id=tokenizer.sep_token_id,
                        )

    trainer = instantiate(cfg.trainer, logger=wandb_logger)

    # Train the model
    trainer.fit(model,
                datamodule=datamodule,
                )

    # load the best model
    model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"])

    # Save the model in the output directory with the project name and current date and time
    output_path = Path(cfg.output_dir)
    timedate = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_path / "_".join([cfg.project_name, timedate])
    vision_encoder_path = output_path / "vision_encoder"
    text_encoder_path = output_path / "text_encoder"
    vision_encoder_path.mkdir(parents=True)
    text_encoder_path.mkdir(parents=True)

    model.vision_model.save_pretrained(vision_encoder_path)
    model.text_model.save_pretrained(text_encoder_path)
    tokenizer.save_pretrained(text_encoder_path)

    # Log path to the saved model
    wandb_logger.log_hyperparams({"model_output_path": str(output_path)})


if __name__ == "__main__":
    main()
