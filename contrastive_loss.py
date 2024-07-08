import torch
import torch.nn as nn

# from timm.models.vision_transformer import PatchEmbed, Block


import numpy as np


class PatchEmbeding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, dropout=0., norm_layer=None):
        super().__init__()
        self.patchsize = patch_size
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # [n, c, h, w]
        B, C, H, W = x.shape
        x = self.patch_embedding(x)  # [n, c', h', w']
        x = x.flatten(2)  # [n, c', h'*w']
        x = x.transpose(1, 2)  # [n, h'*w', c']
        x = self.dropout(x)
        x = self.norm(x)
        return x

class CL(nn.Module):
    def __init__(self,in_C,out_C ):
        super().__init__()
        # --------------------------------------------------------------------------
        # encoder
        self.patch_embed = PatchEmbeding(patch_size=4, in_channels=in_C, embed_dim=out_C,dropout=0.)
        self.patch_embed2 = PatchEmbeding(patch_size=4, in_channels=in_C, embed_dim=out_C,dropout=0.)

        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth)])

        # self.norm_pix_loss = norm_pix_loss

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_hha = nn.CrossEntropyLoss()


        self.initialize_weights()

    def initialize_weights(self):

        w = self.patch_embed.patch_embedding.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        w2 = self.patch_embed2.patch_embedding.weight.data
        torch.nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
    #
        # initialize nn.Linear and nn.LayerNorm
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         # we use xavier_uniform following official JAX ViT:
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)


    def forward(self, imgs, hha):

        hha = self.patch_embed2(hha)
        imgs = self.patch_embed(imgs)


        # for blk in self.blocks:
        #     hha = blk(hha)
        #     imgs = blk(imgs)

        # normalized features
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        hha = hha / hha.norm(dim=-1, keepdim=True)
        loss_all = 0
        for i in range(0, imgs.size(0)):
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * imgs[i] @ hha[i].t()
            logits_per_hha = logits_per_image.t()
            labels = torch.arange(imgs.size(1), dtype=torch.long).cuda()
            loss_i = self.loss_img(logits_per_image, labels)
            loss_h = self.loss_hha(logits_per_hha, labels)
            loss = (loss_i + loss_h) / 2
            loss_all = loss_all + loss
        loss_avg = loss_all / imgs.size(0)

        return loss_avg