import torch
import torch.nn as nn
from .backbones import Backbones
from .layers import MHSA, PositionalEncoding

class MQFusionNet(nn.Module):
    def __init__(self, num_classes=14, query_dim=256, query_len=64):
        super(MQFusionNet, self).__init__()
        
        # --- 1. Heterogeneous Backbones ---
        self.vgg_features = Backbones.get_vgg16()
        self.resnet_features = Backbones.get_resnet50()
        self.densenet_features = Backbones.get_densenet121()
        
        # --- 2. Shared Encoder & Alignment (Stage 1) ---
        # 1x1 Convolutions for dimensionality reduction (Eq. 2)
        self.proj_vgg = nn.Conv2d(512, query_dim, kernel_size=1)
        self.proj_res = nn.Conv2d(2048, query_dim, kernel_size=1)
        self.proj_dense = nn.Conv2d(1024, query_dim, kernel_size=1)
        
        # Task-specific Queries Initialization (Eq. 3)
        self.query_len = query_len
        self.query_embed = nn.Parameter(torch.randn(3, query_len, query_dim))
        self.pos_enc = PositionalEncoding(query_dim, query_len)
        
        # Feature-Query Interaction (Eq. 4-5)
        self.mhsa_vgg = MHSA(query_dim)
        self.mhsa_res = MHSA(query_dim)
        self.mhsa_dense = MHSA(query_dim)
        
        # --- 3. Cross-Net Query Attention Module (Stage 2) ---
        # Inter-network dependencies (Eq. 7)
        self.cross_net_attn = MHSA(query_dim)
        self.mlp_global = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim)
        )
        
        # Dynamic Weight Generator (Eq. 9)
        self.weight_gen = nn.Sequential(
            nn.Linear(query_dim, 1)
        )
        
        # --- 4. Shared Decoder (Stage 3) ---
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        
        # Final Classification Head (Eq. 11-12)
        self.classifier = nn.Sequential(
            nn.Conv2d(query_dim * 3, 512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Step 1: Extract Features
        f_vgg = self.vgg_features(x)
        f_res = self.resnet_features(x)
        f_dense = self.densenet_features(x)
        
        # Step 2: Align Features
        x_vgg = self.proj_vgg(f_vgg)
        x_res = self.proj_res(f_res)
        x_dense = self.proj_dense(f_dense)
        
        # Flatten [B, C, H, W] -> [B, H*W, C] for Attention
        flat_vgg = x_vgg.flatten(2).permute(0, 2, 1)
        flat_res = x_res.flatten(2).permute(0, 2, 1)
        flat_dense = x_dense.flatten(2).permute(0, 2, 1)
        
        # Prepare Queries
        q_vgg = self.pos_enc(self.query_embed[0].unsqueeze(0).expand(batch_size, -1, -1))
        q_res = self.pos_enc(self.query_embed[1].unsqueeze(0).expand(batch_size, -1, -1))
        q_dense = self.pos_enc(self.query_embed[2].unsqueeze(0).expand(batch_size, -1, -1))
        
        # Interaction
        p_vgg = self.mhsa_vgg(q_vgg, flat_vgg, flat_vgg)
        p_res = self.mhsa_res(q_res, flat_res, flat_res)
        p_dense = self.mhsa_dense(q_dense, flat_dense, flat_dense)
        
        # Step 3: Cross-Net Attention
        P_cat = torch.cat([p_vgg, p_res, p_dense], dim=1)
        P_cat_prime = self.cross_net_attn(P_cat, P_cat, P_cat)
        P_cat_prime = P_cat_prime + self.mlp_global(P_cat_prime)
        
        # Split back
        p_vgg_prime = P_cat_prime[:, :self.query_len, :]
        p_res_prime = P_cat_prime[:, self.query_len:2*self.query_len, :]
        p_dense_prime = P_cat_prime[:, 2*self.query_len:, :]
        
        # Compute Dynamic Weights (Eq. 9)
        w_vgg = self.weight_gen(p_vgg_prime.mean(dim=1))
        w_res = self.weight_gen(p_res_prime.mean(dim=1))
        w_dense = self.weight_gen(p_dense_prime.mean(dim=1))
        
        alpha = torch.softmax(torch.cat([w_vgg, w_res, w_dense], dim=1), dim=1)
        
        # Step 4: Weighted Fusion & Decoding
        # Apply weights
        x_vgg_w = x_vgg * alpha[:, 0].view(-1, 1, 1, 1)
        x_res_w = x_res * alpha[:, 1].view(-1, 1, 1, 1)
        x_dense_w = x_dense * alpha[:, 2].view(-1, 1, 1, 1)
        
        # Upsample and Concat
        X_final = torch.cat([
            self.upsample(x_vgg_w),
            self.upsample(x_res_w),
            self.upsample(x_dense_w)
        ], dim=1)
        
        # Predict
        logits = self.classifier(X_final)
        
        return logits, alpha