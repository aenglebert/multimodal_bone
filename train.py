import torch
import numpy as np
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

import datasets

from accelerate.state import DistributedType

import webdataset as wds

import os
from pathlib import Path

from data import StudyCollator, BatchedWebLoaderLen

import time

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


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_train_dataloader(self):
        """
               Returns the training [`~torch.utils.data.DataLoader`].

               Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
               training if necessary) otherwise.

               Subclass and override this method if you want to inject some custom behavior.
               """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        if isinstance(train_dataset, wds.compat.WebDataset):
            dataloader_params["collate_fn"] = lambda x: x
            dataloader = self.accelerator.prepare(
                BatchedWebLoaderLen(train_dataset.batched(self._train_batch_size, collation_fn=data_collator), **dataloader_params))
            return dataloader

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


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

    # set distributed_state manually
    if "deepspeed" in cfg.training_args and cfg.training_args.deepspeed:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # Instantiate the model
    model = instantiate(cfg.vlp_model, vision_model=image_encoder, text_model=text_encoder)

    # Instantiate the trainer
    trainer = CustomTrainer(model=model,
                      args=training_args,
                      train_dataset=dataset,
                      #eval_dataset=tokenized_dataset["test"],
                      data_collator=data_collator,
                      )

    # Train the model
    #trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from_checkpoint)
    trainer.train()

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


if __name__ == "__main__":
    main()
