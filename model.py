from collections import OrderedDict

import torch
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
def drop_path(x, drop_prob: float = 0., training: bool = False):
    # eval mode, dont drop
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    con2D or linear projection
    """
    def __init__(self, img_size=224, patch_size=16, in_channel=3, embed_dim=768, projection="con2d", norm_layer=None):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        # self.grid_size = (self.img_size[0] // self.patch_size[0],
        #                   self.img_size[0] // self.patch_size[0])
        # self.num_patch = self.grid_size[0] * self.grid_size[1]
        # initiate class info and position info, can be learnt
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))  # [197 * 768]
        if projection == "con2d":
            #  [B, C, H, W] -> [B, C, HW]
            #  [B, C, HW] -> [B, HW, C]
            self.projection = nn.Sequential(
                nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e h w -> b (h w) e')
            )
        elif projection == "linear":
            self.projection = nn.Sequential(
                Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
                nn.Linear(patch_size * patch_size * in_channel, embed_dim)
            )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)  # concatenate class info
        x += self.positions  # embed position info
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, atten_drop_prob=0., proj_drop_prob=0.):
        super(MultiHeadAttention, self).__init__()
        # Initiate some parameters
        self.num_heads = num_heads  # number of attention head
        head_dim = dim // num_heads  # dim: dimension of token(embedding dim);
                                     # head_dim: dimension of each attention head
        self.scale = qk_scale or head_dim ** -0.5  # alpha = q * k / sqrt(d), where d is head dimension

        # initiate some transformations
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_drop = nn.Dropout(atten_drop_prob)
        self.projection = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_prob)

    def forward(self, x):
        # x shape -> [batch_size, num_patches + 1, total_embed_dim], _ * 197 * 768
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim], _ * 197 * (3*768)
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head], _ * 197 * 3 * 12 * 64
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head], 3 * _ * 12 * 197 * 64
        qkv = rearrange(self.qkv(x), 'b n (qkv h d) -> (qkv) b h n d', h=self.num_heads, qkv=3)

        # q,k,v -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head], _ * 12 * 197 * 64
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1], _ * 12 * 64 * 197
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # [_ * 12 * 197 * 64] @ [_ * 12 * 64 * 197] -> [_ * 12 * 197 * 197]
        alpha = (q @ k.transpose(-2, -1)) * self.scale

        attention = alpha.softmax(dim=-1)
        attention = self.attention_drop(attention)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # [_ * 12 * 197 * 197] @ [_ * 12 * 197 * 64] -> [_ * 12 * 197 * 64]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim], [_ * 197 * 768]
        x = attention @ v
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.projection(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_prob=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.drop1 = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_expand: int=4, qkv_bias=False, qk_scale=None,
                 drop_prob=0., atten_drop_prob=0., drop_path_prob=0.):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            atten_drop_prob=atten_drop_prob, proj_drop_prob=drop_prob)
        # self.attn = nn.MultiheadAttention(dim, num_heads) pytorch中已经集成了注意力机制
        self.drop = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Dropout(drop_prob)
        self.norm2 = nn.LayerNorm(dim)

        # generally, expand 4 times and then narrow back by setting "out_feature" None
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_expand, drop_prob=drop_prob)

    def forward(self, x):
        x = x + self.drop(self.attention(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_expand: int = 4, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_prob=0., atten_drop_prob=0.,
                 drop_path_prob=0., classifier_mode="token", embed_layer=PatchEmbed):
        super().__init__()
        self.classifier_mode = classifier_mode
        assert classifier_mode == "token" or classifier_mode == "gap", \
                                                      f"classifier mode only includes \"token\" and \"gap\"."
        self.num_classes = num_classes  # need to be classified
        self.embed_dim = self.num_features = embed_dim  # the dimension after embedding
        # self.num_tokens = 2 if distilled else 1  # [optional] model includes a distillation token
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                        in_channel=in_channels, embed_dim=embed_dim)
        self.dropout = nn.Dropout(p=drop_prob)

        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, depth)]
        self.encoderblock = nn.Sequential(*[EncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_expand=mlp_expand,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop_prob=drop_prob,
                                                        atten_drop_prob=atten_drop_prob, drop_path_prob=dpr[i])
                                            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier Head
        # pre-logit: -> a fully connected layer and active function
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("linear", nn.Linear(embed_dim, representation_size)),
                ("tanh", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        # linear
        self.classifier = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)  # [b, 3, 224, 224] -> [b, 3, 197, 168]
        x = self.dropout(x)
        x = self.encoderblock(x)  # [b, 3, 197, 168] -> [b, 197, 768]
        x = self.norm(x)
        if self.classifier_mode == "token":
            x = x[:, 0]
        elif self.classifier_mode == "gap":
            x = reduce(x, 'b n e -> b e', reduction='mean')
        else:
            raise ValueError(f'Invalid classifier={self.classifier_mode}')
        x = self.pre_logits(x)
        x = self.classifier(x)
        return x


# def vit_base_patch16_224(num_classes: int = 1000):
#     model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=768,
#                               depth=12,
#                               num_heads=12,
#                               representation_size=None,
#                               num_classes=num_classes)
#     return model
#
#
# def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
model = VisionTransformer(img_size=224,
                          patch_size=16,
                          embed_dim=768,
                          depth=12,
                          num_heads=12,
                          representation_size=768,
                          num_classes=2,
                          classifier_mode='token')
#     return model


summary(model, (3, 224, 224), device="cpu")

