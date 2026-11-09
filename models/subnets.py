"""Backbone encoders and cross-modal interaction modules.

`BERTEncoder` encodes the literal text and the distilled Core Description
through a shared BERT-large backbone. `BertCrossEncoder` implements the
anchor-guided cross-attention that re-expresses video / audio from the
viewpoint of a query anchor (literal text or semantic description).
"""
import copy
import math
import torch
from torch import nn
from transformers import BertModel

__all__ = ['BERTEncoder', 'BertCrossEncoder']


class BERTEncoder(nn.Module):

    def __init__(self, args):
        super(BERTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(args.text_backbone_path)
        # Gradient checkpointing: text / desc share the backbone, so keeping both
        # activation graphs resident would otherwise exhaust GPU memory.
        self.bert.config.use_cache = False
        self.bert.gradient_checkpointing_enable()

    def forward(self, text_feats):
        input_ids, input_mask, segment_ids = text_feats[:, 0], text_feats[:, 1], text_feats[:, 2]
        outputs = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        return outputs.last_hidden_state


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size=1024):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size * 4)

    def forward(self, hidden_states):
        return gelu(self.dense(hidden_states))


class BertCoAttention(nn.Module):
    def __init__(self, num_attention_heads=8, hidden_size=1024, dp_rate=0.1):
        super(BertCoAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dp_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertOutput(nn.Module):
    def __init__(self, hidden_size=1024, dp_rate=0.1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(hidden_size * 4, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size=1024, dp_rate=0.1):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dp_rate):
        super(BertCrossAttention, self).__init__()
        self.bertCoAttn = BertCoAttention(num_attention_heads, hidden_size, dp_rate)
        self.output = BertSelfOutput(hidden_size, dp_rate)

    def forward(self, s1_input_tensor, s2_input_tensor):
        s1_cross_output = self.bertCoAttn(s1_input_tensor, s2_input_tensor)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dp_rate):
        super(BertCrossAttentionLayer, self).__init__()
        self.bertCorssAttn = BertCrossAttention(num_attention_heads, hidden_size, dp_rate)
        self.intermediate = BertIntermediate(hidden_size)
        self.output = BertOutput(hidden_size, dp_rate)

    def forward(self, s1_hidden_states, s2_hidden_states):
        attention_output = self.bertCorssAttn(s1_hidden_states, s2_hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertCrossEncoder(nn.Module):
    """Stacked cross-attention encoder: re-expresses `s1` from the viewpoint of `s2`.

    Used four times in CESF, once per (anchor, non-text-modality) pair:
      - literal anchor  : text -> video, text -> audio
      - semantic anchor : desc -> video, desc -> audio
    """
    def __init__(self, num_attention_heads, hidden_size, dp_rate, n_layers):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(num_attention_heads, hidden_size, dp_rate)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, s1_hidden_states, s2_hidden_states):
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states)
        return s1_hidden_states
