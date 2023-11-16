from typing import Optional, Tuple

import torch

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.configuration_utils import PretrainedConfig

from models.seq_vision_transformer import SeqVisionTransformer

AUTO_MAP = {
    "AutoModel": "modeling_seqvit_model.SeqViTEncoder",
}


class SeqViTConfig(PretrainedConfig):
    def __init__(self, model_name='vit_base_patch16_224', state_dict=None, strict_load=True, **kwargs):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.state_dict = state_dict
        self.strict_load = strict_load
        self.hidden_size = None


class SeqViTEncoder(PreTrainedModel):
    config_class = SeqViTConfig
    base_model_prefix = "vit_base_patch16_224"

    def __init__(
            self,
            config: Optional[SeqViTConfig] = None
    ):
        if config is None:
            config = SeqViTConfig()
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        vit = SeqVisionTransformer()

        # Get hidden size from the model
        config.hidden_size = vit.head.in_features

        # initialize with config
        super().__init__(config)

        # set the model
        self.vit = vit

        # remove the head
        self.vit.head = torch.nn.Identity()

        # load the state dict if pretrained model
        if config.state_dict:
            self.vit.load_state_dict(torch.load(config.state_dict), strict=config.strict_load)

        self.vit.norm = torch.nn.Identity()

    def forward(
            self,
            pixel_values=None,
    ):

        x = pixel_values

        if x.dim() == 5:
            bs = x.shape[0]
            seq_len = x.shape[1]
            x = x.flatten(0, 1).contiguous()
        elif x.dim() == 4:
            bs = x.shape[0]
            seq_len = 1
        else:
            raise ValueError(f'Input tensor must be 4D or 5D, got: {x.dim()}')

        x = self.vit.forward_features(x, seq_len=seq_len)

        # reshape to B,S,Cout if seq_len > 1
        if seq_len > 1:
            x = x.view(bs, seq_len, x.shape[1], -1)

            # Get CLS tokens
            cls_token = x[:, :, 0].mean(1)
        else:
            cls_token = x[:, 0]

        return BaseModelOutputWithPooling(
            last_hidden_state=x,
            pooler_output=cls_token,
            hidden_states=None,
            attentions=None,
        )
