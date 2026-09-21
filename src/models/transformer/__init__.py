import torch.nn as nn
import torch.nn.functional as F

from models.transformer.decoder import TransformerDecoder
from models.transformer.encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.0):
        super().__init__()
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, dropout)

    def forward(self, src, src_mask, tgt, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(enc_out, src_mask, tgt, tgt_mask)
        return out


class ResidualRoleAdapter(nn.Module):
    """Small modality-specific residual used around a shared Transformer."""

    def __init__(self, d_model, bottleneck):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        # Weight-merging initialization starts the merged branch as an identity map.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.norm(x))))


class DualTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_decoder_layers1, num_decoder_layers2,
                 dropout=0.0, share_decoders=False, role_adapter_dim=0,
                 capture_intermediate=False):
        super().__init__()
        self.share_decoders = share_decoders
        if share_decoders:
            if num_decoder_layers1 != num_decoder_layers2:
                raise ValueError("shared decoder requires equal directional depths")
            self.shared_decoder = TransformerDecoder(num_decoder_layers1, d_model, num_heads, dropout)
            adapter_dim = role_adapter_dim or max(16, d_model // 8)
            self.video_adapter = ResidualRoleAdapter(d_model, adapter_dim)
            self.text_adapter = ResidualRoleAdapter(d_model, adapter_dim)
        else:
            self.decoder1 = TransformerDecoder(num_decoder_layers1, d_model, num_heads, dropout)
            self.decoder2 = TransformerDecoder(num_decoder_layers2, d_model, num_heads, dropout)
        decoders = ([self.shared_decoder] if share_decoders else
                    [self.decoder1, self.decoder2])
        for decoder in decoders:
            decoder.capture_hidden = capture_intermediate

    def set_progressive_drop(self, layer_index=None, keep_ratio=1.0):
        decoders = ([self.shared_decoder] if self.share_decoders else
                    [self.decoder1, self.decoder2])
        for decoder in decoders:
            decoder.progressive_drop_index = layer_index
            decoder.progressive_keep_ratio = float(keep_ratio)

    def forward(self, src1, src_mask1, src2, src_mask2, decoding, enc_out=None, gauss_weight=None, need_weight=False):
        assert decoding in [1, 2]
        if self.share_decoders:
            video = self.video_adapter(src1)
            text = self.text_adapter(src2)
            if decoding == 1:
                if enc_out is None:
                    enc_out, _ = self.shared_decoder(None, None, text, src_mask2)
                out, weight = self.shared_decoder(enc_out, src_mask2, video, src_mask1)
            else:
                if enc_out is None:
                    enc_out, _ = self.shared_decoder(
                        None, None, video, src_mask1, tgt_gauss_weight=gauss_weight)
                out, weight = self.shared_decoder(
                    enc_out, src_mask1, text, src_mask2, src_gauss_weight=gauss_weight)
        else:
            if decoding == 1:
                if enc_out is None:
                    enc_out, _ = self.decoder2(None, None, src2, src_mask2)
                out, weight = self.decoder1(enc_out, src_mask2, src1, src_mask1)
            else:
                if enc_out is None:
                    enc_out, _ = self.decoder1(None, None, src1, src_mask1, tgt_gauss_weight=gauss_weight)
                out, weight = self.decoder2(enc_out, src_mask1, src2, src_mask2, src_gauss_weight=gauss_weight)
        
        if need_weight:
            return enc_out, out, weight
        return enc_out, out
