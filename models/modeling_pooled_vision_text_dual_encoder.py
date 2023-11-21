from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoModel

from .configuration_pooled_vision_text_dual_encoder import PooledVisionTextDualEncoderConfig


class SigLIPLoss(nn.Module):
    def __init__(self, temperature: float = 10.0, bias: float = -10.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, similarity: torch.Tensor) -> torch.Tensor:
        logits = similarity * self.temperature + self.bias
        n = len(logits)
        labels = 2 * torch.eye(n, device=logits.device) - 1
        return -torch.sum(nn.functional.logsigmoid(labels * logits)) / n


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class OrthoOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
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

    loss: Optional[torch.FloatTensor] = None
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
    ):
        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        if config is None:
            config = PooledVisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

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
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained from
            the text model.

        Examples:

        ```python
        >>> from transformers import VisionTextDualEncoderModel, AutoTokenizer

        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        >>> tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")

        >>> inputs = tokenizer(["una foto di un gatto", "una foto di un cane"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
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

        vision_outputs = self.vision_model(pixel_values)

        image_features = vision_outputs[1]  # pooled_output

        return image_features

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            seq_attr: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            doc_embeddings: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> Union[Tuple[torch.Tensor], OrthoOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vision_outputs = self.vision_model(pixel_values)
        image_embeds = vision_outputs[1]  # pooled_output

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = self.loss(logits_per_text)
        if not loss > 0:
            print("loss", loss)
            print("text_embeds", text_embeds)
            print("image_embeds", image_embeds)
            print("seq_attr", seq_attr.p)
            print("logits_per_text", logits_per_text)
            print("logits_per_image", logits_per_image)
            print("logit_scale", logit_scale)

        if not return_dict:
            return loss, logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs

        return OrthoOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
