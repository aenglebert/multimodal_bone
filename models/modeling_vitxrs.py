from typing import Dict, List, Optional, Set, Tuple, Union

import math

from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import ViTConfig, ViTModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput, ModelOutput
from transformers.configuration_utils import PretrainedConfig

from transformers.models.vit.modeling_vit import ViTAttention, ViTIntermediate, ViTOutput, ViTEmbeddings, ViTPooler
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEEmbeddings

from models.seq_vision_transformer import SeqVisionTransformer

AUTO_MAP = {
    "AutoModel": "modeling_vitxrs.ViTXRSModel",
}


class ViTXRSConfig(ViTConfig):

    model_type = "vitxrs"

    def __init__(self, pooling="mean_cls_token", num_cls_attention_heads=2, **kwargs):
        super().__init__(**kwargs)

        self.num_cls_attention_heads = num_cls_attention_heads
        self.seq_model = True
        self.pooling = pooling
        self.mask_ratio = 0


@dataclass
class ViTXRSModelOutput(ModelOutput):
    """
    Class for ViTXRSModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, token_sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        grouped_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, images_sequence_lenght, token_sequence_length, hidden_size)`):
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ViTXRSPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CLSSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_cls_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_cls_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_cls_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ViTXRSCLSAttention(nn.Module):
    """Attention between the cls tokens inside of a sequence of images."""
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.attention = CLSSelfAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        if seq_len is None:
            seq_len = 1

        # Get CLS tokens for the batch
        cls_tokens = hidden_states[:, 0, :].clone()  # B x C

        # Group CLS by sequence length
        cls_tokens = cls_tokens.view(-1, seq_len, cls_tokens.shape[-1])  # (B * S) x S x C

        # Compute attention over CLS tokens
        cls_tokens = cls_tokens + self.attention(cls_tokens)[0]

        # Reshape back to batch
        cls_tokens = cls_tokens.view(-1, cls_tokens.shape[-1])  # B x C

        # Update input_dict
        hidden_states[:, 0, :] = cls_tokens

        return (hidden_states, )


class ViTRXSLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.cls_attention = ViTXRSCLSAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # Attention between CLS tokens
        attention_output = self.cls_attention(
            attention_output,
            seq_len,
            output_attentions=output_attentions,
        )[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTRXSEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTRXSLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    seq_len,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, seq_len, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTXRSModel(ViTModel):
    config_class = ViTXRSConfig

    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTMAEEmbeddings(config)
        self.encoder = ViTRXSEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ViTXRSModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if pixel_values.dim() == 5:
            bs = pixel_values.shape[0]
            seq_len = pixel_values.shape[1]
            pixel_values = pixel_values.flatten(0, 1).contiguous()

        elif pixel_values.dim() == 4:
            bs = pixel_values.shape[0]
            seq_len = 1
        else:
            raise ValueError(f'Input tensor must be 4D or 5D, got: {pixel_values.dim()}')

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output, mask, ids_restore = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            seq_len=seq_len,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if seq_len > 1:
            grouped_sequence_output = sequence_output.view(bs, seq_len, sequence_output.shape[1], -1)
            pooled_cls_token = grouped_sequence_output[:, :, 0].mean(1)

        else:
            pooled_cls_token = sequence_output[:, 0]

        pooled_output = self.pooler(pooled_cls_token.unsqueeze(1)) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return ViTXRSModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
