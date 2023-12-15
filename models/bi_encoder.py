import torch
from pytorch_lightning import LightningModule

import bitsandbytes as bnb

from .utils import ClipLoss, SigLIPLoss, optimizer_dict


class BiEncoder(LightningModule):
    def __init__(self,
                 vision_model=None,
                 text_model=None,
                 global_loss_fn=ClipLoss(),
                 local_loss_fn=None,
                 optimizer="AdamW",
                 lr=1e-5,
                 **kwargs):
        super().__init__()

        assert vision_model is not None and text_model is not None, "Vision and text models must be provided"

        self.save_hyperparameters(ignore=["vision_model", "text_model", "global_loss_fn", "local_loss_fn"])

        self.vision_model = vision_model
        self.text_model = text_model
        self.global_loss_fn = global_loss_fn
        self.local_loss_fn = local_loss_fn
        self.optimizer = optimizer

    def common_step(self, batch, batch_idx):
        input_ids = batch["input_ids"] if "input_ids" in batch else None
        pixel_values = batch["pixel_values"] if "pixel_values" in batch else None
        pooling_matrix = batch["pooling_matrix"] if "pooling_matrix" in batch else None
        attention_mask = batch["attention_mask"] if "attention_mask" in batch else None
        position_ids = batch["position_ids"] if "position_ids" in batch else None
        token_type_ids = batch["token_type_ids"] if "token_type_ids" in batch else None
        output_attentions = batch["output_attentions"] if "output_attentions" in batch else None
        return_dict = batch["return_dict"] if "return_dict" in batch else None
        image_text_pairs = batch["image_text_pairs"] if "image_text_pairs" in batch else None

        if hasattr(self.vision_model.config, 'seq_model') and self.vision_model.config.seq_model:
            # The vision model is a sequence model, it handles the pooling internally
            vision_outputs = self.vision_model(pixel_values=pixel_values,
                                               pooling_matrix=pooling_matrix,
                                               )
            images_global_features = vision_outputs.pooler_output
            images_global_proj = self.vision_model.projection(images_global_features)

        else:
            # The vision model is not a sequence model, we need to handle the pooling
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            images_global_features = vision_outputs.last_hidden_state[:, 0, :]
            images_global_proj = self.vision_model.projection(images_global_features)

        images_local_features = vision_outputs.last_hidden_state[:, 1:, :]
        images_local_proj = self.vision_model.projection(images_local_features)

        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "output_attentions": output_attentions,
            "return_dict": return_dict,
        }

        if "labels" in batch:
            text_inputs["labels"] = batch["labels"]

        text_outputs = self.text_model(
            **text_inputs
        )

        text_proj = self.text_model.projection(text_outputs.last_hidden_state)
        text_global_proj = text_proj[:, 0, :]
        text_local_proj = text_proj[:, 1:, :]

        # normalizing the projection vectors
        images_local_proj = images_local_proj / images_local_proj.norm(dim=-1, keepdim=True)
        images_global_proj = images_global_proj / images_global_proj.norm(dim=-1, keepdim=True)
        text_local_proj = text_local_proj / text_local_proj.norm(dim=-1, keepdim=True)
        text_global_proj = text_global_proj / text_global_proj.norm(dim=-1, keepdim=True)

        # Compute global similarity
        global_sim = torch.matmul(images_global_proj, text_global_proj.T)

        # Compute global loss
        global_loss = self.global_loss_fn(global_sim)

        # TODO: Compute local similarity and loss

        return global_loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optimizer_dict[self.optimizer](self.parameters(), lr=self.hparams.lr)
        # Freeze embedding layer and 6 first layers of the text model
        self.text_model.embeddings.requires_grad_(False)
        for param in self.text_model.encoder.layer[:6].parameters():
            param.requires_grad = False
        return optimizer
