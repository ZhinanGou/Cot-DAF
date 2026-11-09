"""CESF: Comprehensive Semantic-augmented shallow-to-deep Fusion.

Dual-anchor, multi-granularity fusion for multimodal intent recognition.

The model consumes four inputs -- literal text, video, audio, and a distilled
Core Description produced by an offline semantic-distillation stage -- and
predicts the
speaker's communicative intent. Its two contributions over prior work (SDIF,
CESF/9130):

  1. *Dual anchors*. Two guidance signals are used instead of one:
       - literal anchor:  text -> video / audio  (raw wording)
       - semantic anchor: desc -> video / audio  (audio-visual conflict arbitration)
     Four cross-attention operations re-express the non-text modalities from
     each anchor's viewpoint.

  2. *Six-granularity rep-level deep interaction*. Five masked-mean pooled reps
     (text, v2t, a2t, v2d, a2d) plus one shallow-fused rep are stacked and fed
     through a Transformer encoder so the granularities can reconcile the
     divergence between the literal and semantic anchors -- replacing the
     black-box gate of the baseline with explicit, learnable arbitration.

  3. *Capacity-controlled classifier*. A bottleneck projection + dropout
     regularizes the high-dimensional fusion vector (6 x 1024), which turns the
     accuracy gain from statistically insignificant into significant.

The raw unimodal video / audio reps and the standalone desc rep are *deliberately
omitted*: the distillation stage already compressed the intent-relevant
audio-visual semantics into the Core Description, so the low-level non-text
reps and the
text-form desc rep (redundant with the literal text under the same BERT
backbone) add little. The value of the description lies in using it as a
*cross-modal semantic anchor*, not in appending it as extra text.
"""
import torch
from torch import nn
from .subnets import BERTEncoder, BertCrossEncoder


class CESF(nn.Module):

    def __init__(self, args):
        super(CESF, self).__init__()

        self.text_feats_dim = args.text_feats_dim
        self.video_feats_dim = args.video_feats_dim
        self.audio_feats_dim = args.audio_feats_dim
        self.hidden_feats_dim = args.text_feats_dim

        self.cross_num_heads = args.cross_num_heads
        self.dst_feature_dims = args.dst_feature_dims
        self.cross_dp_rate = args.cross_dp_rate
        self.layers_cross = args.n_levels_cross
        self.layers_self = args.n_levels_self
        self.self_num_heads = args.self_num_heads

        self.weight_dropout = args.weight_dropout
        self.activation = nn.ReLU()
        self.num_labels = args.num_labels

        # Shared BERT backbone for the two text-form inputs (utterance + desc).
        self.text_subnet = BERTEncoder(args)

        # Project video / audio into the text dimension.
        self.video_hidden_net = nn.Sequential(
            nn.Linear(self.video_feats_dim, self.hidden_feats_dim),
            nn.ReLU()
        )
        self.audio_hidden_net = nn.Sequential(
            nn.Linear(self.audio_feats_dim, self.hidden_feats_dim),
            nn.ReLU()
        )

        # Shared interaction-space projection for all non-text representations.
        self.share_net = nn.Sequential(
            nn.Linear(self.text_feats_dim, self.text_feats_dim),
            nn.ReLU()
        )

        # Semantic anchor: desc (comprehensive semantics) as query over video / audio.
        self.desc2video_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)
        self.desc2audio_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)

        # Literal anchor: text (raw wording) as query over video / audio.
        self.text2video_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)
        self.text2audio_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)

        # Shallow fusion MLP (compresses the literal-anchor group into a shallow rep).
        self.mlp_project = nn.Sequential(
            nn.Linear(self.dst_feature_dims, self.dst_feature_dims),
            nn.Dropout(self.cross_dp_rate),
            nn.GELU()
        )

        # Rep-level deep interaction: let the six granularities attend to one another.
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dst_feature_dims, nhead=self.self_num_heads)
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=self.layers_self)

        # Six granularities: text, v2t, a2t, v2d, a2d, shallow.
        self.num_reps = 6

        # Capacity-controlled classifier: bottleneck + dropout to curb overfitting.
        self.fusion = nn.Sequential(
            nn.Linear(self.num_reps * self.dst_feature_dims, self.dst_feature_dims),
            nn.Dropout(self.weight_dropout),
            self.activation,
            nn.Linear(self.dst_feature_dims, self.num_labels),
        )

    def _mask_mean(self, seq, mask):
        # seq [B,L,D], mask [B,L] -> [B,D] (valid-length weighted mean).
        m = mask.float().unsqueeze(-1)
        return (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def forward(self, text_feats, video_feats, audio_feats, desc_compre,
                mask_seq=None, label_seq=None, mask_pos=None, label_start=None, label_len=None):

        video_feats = video_feats.float()
        audio_feats = audio_feats.float()

        # 1. Unimodal encoding.
        text_embeds = self.text_subnet(text_feats)               # [B,50,1024]
        video_embeds = self.video_hidden_net(video_feats)        # [B,180,1024]
        audio_embeds = self.audio_hidden_net(audio_feats)        # [B,400,1024]

        text_mask = text_feats[:, 1]                             # [B,50]
        video_mask = (video_feats.abs().sum(dim=-1) > 0).float() # [B,180]
        audio_mask = (audio_feats.abs().sum(dim=-1) > 0).float() # [B,400]

        desc_embeds = self.text_subnet(desc_compre)              # [B,190,1024]
        desc_mask = desc_compre[:, 1]                            # [B,190]
        desc_shared = self.share_net(desc_embeds)

        # 2. Cross-modal interaction space.
        video_shared = self.share_net(video_embeds)
        audio_shared = self.share_net(audio_embeds)

        # 3. Literal anchor (text -> video / audio).
        video_from_text = self.text2video_cross(text_embeds, video_shared)   # [B,50,1024]
        audio_from_text = self.text2audio_cross(text_embeds, audio_shared)   # [B,50,1024]

        # 4. Semantic anchor (desc -> video / audio).
        video_from_desc = self.desc2video_cross(desc_shared, video_shared)   # [B,190,1024]
        audio_from_desc = self.desc2audio_cross(desc_shared, audio_shared)   # [B,190,1024]

        # 5. Six multi-granularity reps (masked mean pooling).
        reps = []
        reps.append(self._mask_mean(text_embeds, text_mask))             # text_rep
        reps.append(self._mask_mean(video_from_text, text_mask))         # video2text_rep
        reps.append(self._mask_mean(audio_from_text, text_mask))         # audio2text_rep
        reps.append(self._mask_mean(video_from_desc, desc_mask))         # video2desc_rep
        reps.append(self._mask_mean(audio_from_desc, desc_mask))         # audio2desc_rep

        shallow_seq = self.mlp_project(torch.cat([video_from_text, audio_from_text, text_embeds], dim=1))  # [B,150,1024]
        shallow_mask = torch.cat([text_mask, text_mask, text_mask], dim=-1)                                # [B,150]
        reps.append(self._mask_mean(shallow_seq, shallow_mask))           # shallow_rep

        # 6. Rep-level deep interaction.
        all_reps = torch.stack(reps, dim=0)                              # [6,B,1024]
        all_hiddens = self.self_attention(all_reps)                      # [6,B,1024]
        deep_rep = torch.cat([all_hiddens[i] for i in range(self.num_reps)], dim=-1)  # [B,6*1024]

        logits = self.fusion(deep_rep)                                   # [B,num_labels]

        return {'logits': logits}
