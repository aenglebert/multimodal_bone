from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, Dict

import torch
from torch import nn

from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoModel

from .configuration_pooled_vision_text_dual_encoder import PooledVisionTextDualEncoderConfig

from .utils import clip_loss, SigLIPLoss


@dataclass
class OrthoOutput(ModelOutput):
    """
    Args:
        loss (`Dict(str, :obj:`torch.FloatTensor` of shape `(1,)`, `optional`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained from the text model.
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained from the vision model.
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[Dict[str, torch.FloatTensor]] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class PooledVisionTextDualEncoderModel(PreTrainedModel):
    config_class = PooledVisionTextDualEncoderConfig
    base_model_prefix = "pooled_vision_text_dual_encoder"

    def __init__(
            self,
            config: Optional[PooledVisionTextDualEncoderConfig] = None,
            vision_model: Optional[PreTrainedModel] = None,
            text_model: Optional[PreTrainedModel] = None,
            pool_image: Optional[bool] = None,
    ):
        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        if config is None:
            config = PooledVisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        if pool_image is not None:
            config.pool_image = pool_image

        # initialize with config
        super().__init__(config)

        if vision_model is None:
            vision_model = AutoModel.from_config(config.vision_config)

        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)

        self.vision_model = vision_model
        self.text_model = text_model

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size

        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        if config.loss_type == "siglip":
            self.loss = SigLIPLoss()
        elif config.loss_type == "clip":
            self.loss = clip_loss
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}, expected one of ['siglip', 'clip']")

    def get_text_features(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            token_type_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_features = text_outputs[1]

        # text_features = text_outputs
        return text_features

    def get_image_features(
            self,
            pixel_values=None,
            **kwargs,
    ):

        vision_outputs = self.vision_model(pixel_values, output_hidden_states=True)

        image_features = vision_outputs.pooler_output

        return image_features

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            images_attention_mask: Optional[torch.FloatTensor] = None,
            pooling_matrix: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            image_text_pairs: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[Tuple[torch.Tensor], OrthoOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if image_text_pairs is None:
            image_text_pairs = torch.eye(input_ids.shape[0], device=input_ids.device)

        # If the model handles the pooling internally
        if hasattr(self.vision_model.config, 'seq_model') and self.vision_model.config.seq_model:
            if images_attention_mask is not None:
                vision_outputs = self.vision_model(pixel_values,
                                                   images_attention_mask=images_attention_mask,
                                                   )
            else:
                vision_outputs = self.vision_model(pixel_values,
                                                   pooling_matrix=pooling_matrix,
                                                   )

            image_embeds = vision_outputs.pooler_output

        # Else, we pool the features by a mean pooling using the provided pooling matrix
        elif self.config.pool_image and pooling_matrix is not None:
            vision_outputs = self.vision_model(pixel_values,
                                               )

            image_embeds = vision_outputs.last_hidden_state[:, 0, :]

            image_embeds = pooling_matrix @ image_embeds / pooling_matrix.sum(dim=-1, keepdim=True)

        else:
            vision_outputs = self.vision_model(pixel_values)

            image_embeds = vision_outputs.last_hidden_state[:, 0, :]

            if pooling_matrix is not None:
                image_text_pairs = image_text_pairs @ pooling_matrix

        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "output_attentions": output_attentions,
            "output_hidden_states": True,
            "return_dict": return_dict,
        }

        if "labels" in kwargs:
            text_inputs["labels"] = kwargs.pop("labels")

        text_outputs = self.text_model(
            **text_inputs
        )

        text_cls = text_outputs.hidden_states[-1][:, 0, :]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_cls = text_cls / text_cls.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_cls, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        if self.config.loss_type == "siglip":
            loss = self.loss(logits_per_text, targets=image_text_pairs)

        else:
            loss = self.loss(logits_per_text)

        loss = {"contrastive_loss": loss}

        if "loss" in vision_outputs.keys():
            loss["vision_loss"] = vision_outputs.loss

        if "loss" in text_outputs.keys():
            loss["text_loss"] = text_outputs.loss

        if not return_dict:
            return loss, logits_per_image, logits_per_text, text_cls, image_embeds, text_outputs, vision_outputs

        return OrthoOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_cls,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
