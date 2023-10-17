import torch
import numpy as np
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from transformers import Trainer

import os
from pathlib import Path

from data import StudyCollator

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


@hydra.main(version_base="1.3", config_path="config", config_name="train")
def main(cfg: DictConfig):
    # Set the environment variable for wandb
    os.environ["WANDB_PROJECT"] = cfg.project_name

    # Seed everything
    seed_everything(cfg.seed)

    # Instantiate the dataset
    dataset = instantiate(cfg.dataset)

    # Keep only a part of the dataset for debugging
    #if cfg.debug:
        #dataset = dataset.train_test_split(test_size=0.99)["train"]

    # Keep only training set and split into new train and test sets
    #dataset = dataset.train_test_split(test_size=cfg.test_size)

    # Instantiate the model
    image_encoder = instantiate(cfg.vision_model.encoder)

    tokenizer = instantiate(cfg.text_model.tokenizer)

    text_encoder = instantiate(cfg.text_model.encoder)

    # Instantiate the data collator
    data_collator = StudyCollator(tokenizer=tokenizer)

    # Instantiate the training arguments
    training_args = instantiate(cfg.training_args)

    # Instantiate the model
    model = instantiate(cfg.vlp_model, vision_model=image_encoder, text_model=text_encoder)
    print(model.config)

    # Instantiate the trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=dataset,
                      #eval_dataset=tokenized_dataset["test"],
                      data_collator=data_collator,
                      )

    # Train the model
    #trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from_checkpoint)
    trainer.train()

    # Save the model
#    checkpoint_path = Path(cfg.checkpoint.path)
#    output_path = checkpoint_path.parent / "_".join([checkpoint_path.name] + cfg.dataset.path.split("/"))
#    model.save_pretrained(output_path)


if __name__ == "__main__":
    main()
