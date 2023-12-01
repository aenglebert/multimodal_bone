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

from transformers.utils import is_peft_available
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

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

# Try to import peft
if is_peft_available():
    from peft import PeftModel


# Define a function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_history = {}

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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )

        # Get the loss
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # If multiple losses are returned, log them all and average them
        if isinstance(loss, dict):
            for key, value in loss.items():
                if key not in self.loss_history:
                    self.loss_history[key] = []
                self.loss_history[key].append(value.item())

                if self.state.global_step % self.args.logging_steps == 0 and \
                        len(self.loss_history[key]) > self.args.gradient_accumulation_steps:
                    mean_loss = np.mean(self.loss_history[key])
                    self.log({key: mean_loss, "step": self.state.global_step})
                    self.loss_history[key] = []

            loss = sum(loss.values()) / len(loss)

        return (loss, outputs) if return_outputs else loss


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
