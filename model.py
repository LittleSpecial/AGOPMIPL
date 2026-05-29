#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
RFM-MIPL V12: Dual-Path Attention

New feature: Combines RFM-transformed and raw feature attention
- A_rfm = Attention(H_rfm, H2_rfm)
- A_raw = Attention(H_raw, H2_raw)  
- Final_A = softmax(A_rfm + λ * A_raw)

This increases robustness to noise by preserving raw feature information.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rfm import compute_rfm_sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def forward(self, features, W_U, b_U):
        out = F.linear(features, W_U, b_U)
        out_sigmoid = torch.sigmoid(out)
        return out_sigmoid


class MIPML_V12(nn.Module):
    """
    RFM-MIPL V12: Dual-Path Attention
    
    New parameter:
    - attn_lambda: weight for raw attention path (default: 0.3)
    """
    
    def __init__(self, args):
        super(MIPML_V12, self).__init__()
        self.args = args
        self.L = self.args.L
        self.D = 128
        self.K = 1
        self.nr_fea = self.args.nr_fea
        self.nr_fea_sqrt = math.floor(math.sqrt(self.nr_fea))
        
        # Configurable options
        self.proto_agg = getattr(args, 'proto_agg', 'mean')
        self.inst_weight = getattr(args, 'inst_weight', 0.5)
        self.attn_lambda = getattr(args, 'attn_lambda', 0.3)
        self.bag_repr_mode = getattr(args, 'bag_repr_mode', 'raw')
        self.n_proto_per_class = getattr(args, 'n_proto_per_class', self.args.nr_class)
        self.encoder_type = getattr(args, 'encoder_type', 'default')
        self.transformer_layers = getattr(args, 'transformer_layers', 2)
        self.transformer_heads = getattr(args, 'transformer_heads', 4)
        self.transformer_ffn_mult = getattr(args, 'transformer_ffn_mult', 4)
        self.transformer_dropout = getattr(args, 'transformer_dropout', 0.1)
        
        print(f"[Model V12] Dual-Path Attention")
        print(
            f"  proto_agg={self.proto_agg}, inst_weight={self.inst_weight}, "
            f"attn_lambda={self.attn_lambda}, bag_repr_mode={self.bag_repr_mode}, "
            f"n_proto_per_class={self.n_proto_per_class}, encoder_type={self.encoder_type}"
        )

        # RFM buffers
        self.register_buffer('rfm_matrix', torch.eye(self.L))
        self.register_buffer('rfm_sqrt', torch.eye(self.L))

        # Feature Extractor (no BatchNorm for RFM compatibility)
        is_perfect_square = (self.nr_fea_sqrt * self.nr_fea_sqrt == self.nr_fea)
        self.use_cnn = (self.nr_fea_sqrt >= 8) and is_perfect_square
        
        if self.use_cnn:
            # Wide LeNet: 32/64 channels, padding to preserve spatial info, NO BatchNorm
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),  # 28 -> 28
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),  # 28 -> 14
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14 -> 14
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)  # 14 -> 7
            )
            # Output: 64 * 7 * 7 = 3136
            self.cnn_output_size = 64 * 7 * 7
            print(f"[Model V12] Wide LeNet (32/64), output size: {self.cnn_output_size}")
            
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.cnn_output_size, self.L),
                nn.Dropout(0.4),
                nn.ReLU(),
            )
        else:
            self.feature_extractor_part1 = None
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.nr_fea, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.L),
                nn.ReLU(),
            )
            print(f"[Model V12] FC for {self.nr_fea}-dim input")

        self.sequence_encoder = None
        self.sequence_norm = None
        if self.encoder_type == 'transformer':
            if self.L % self.transformer_heads != 0:
                raise ValueError(
                    f"L={self.L} must be divisible by transformer_heads={self.transformer_heads}"
                )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.L,
                nhead=self.transformer_heads,
                dim_feedforward=self.L * self.transformer_ffn_mult,
                dropout=self.transformer_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False,
            )
            self.sequence_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.transformer_layers,
            )
            self.sequence_norm = nn.LayerNorm(self.L)
            print(
                f"[Model V12] Transformer encoder: layers={self.transformer_layers}, "
                f"heads={self.transformer_heads}, ffn_mult={self.transformer_ffn_mult}"
            )

        # Prototype aggregation for per-class prototypes.
        self.linear1 = nn.Sequential(nn.Linear(self.n_proto_per_class, 1))

        # Shared Attention layers (used for both RFM and raw paths)
        self.att_layer_V = AttentionLayer()
        self.att_layer_U = AttentionLayer()
        self.linear_V = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_U = nn.Linear(self.L * self.K, 1)
        self.attention_weights = nn.Sequential(
            nn.Linear(self.args.nr_class, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K),
        )

        # Classifiers
        classifier_in_dim = self.L if self.bag_repr_mode != 'concat' else self.L * 2
        self.classifier = nn.Linear(classifier_in_dim, self.args.nr_class)
        self.instance_classifier = nn.Linear(self.L, self.args.nr_class)

    def contextualize_sequence(self, H):
        if self.sequence_encoder is None:
            return H
        return self.sequence_norm(self.sequence_encoder(H.unsqueeze(0)).squeeze(0))

    def extract_instance_embeddings(self, X):
        if self.use_cnn:
            if X.dim() == 2:
                X = X.view(-1, 1, self.nr_fea_sqrt, self.nr_fea_sqrt)
            elif X.dim() == 3:
                X = X.unsqueeze(1)
            H = self.feature_extractor_part1(X)
            H = H.view(-1, self.cnn_output_size)
            H = self.feature_extractor_part2(H)
        else:
            H = X.view(X.shape[0], -1)
            H = self.feature_extractor_part2(H)
        return self.contextualize_sequence(H)

    def compute_attention_scores(self, H, H2):
        """Compute raw attention scores (before softmax) using shared attention network"""
        A_V = self.att_layer_V(H, self.linear_V.weight, self.linear_V.bias)
        A_U = self.att_layer_U(H2, self.linear_U.weight, self.linear_U.bias)
        A = self.attention_weights(A_V * A_U.T)
        A = torch.transpose(A, 1, 0)
        return A  # Raw scores, no softmax yet

    def forward(self, X, Z):
        X = X.squeeze(0)
        
        # Feature extraction (raw)
        H_raw = self.extract_instance_embeddings(X)
        
        # RFM Transformation
        H_rfm = H_raw @ self.rfm_sqrt
        
        # Prototype feature extraction
        H2_raw_list = []
        H2_rfm_list = []
        for i in range(len(Z)):
            z_class = Z[i]
            h2_raw = self.extract_instance_embeddings(z_class)
            h2_rfm = h2_raw @ self.rfm_sqrt
            H2_raw_list.append(h2_raw)
            H2_rfm_list.append(h2_rfm)
        
        # Prototype Aggregation
        if self.proto_agg == 'mean':
            H2_raw = torch.stack([torch.mean(h, dim=0) for h in H2_raw_list], dim=0)
            H2_rfm = torch.stack([torch.mean(h, dim=0) for h in H2_rfm_list], dim=0)
        else:
            # Learn a per-class weighted summary over that class's prototypes.
            H2_tensor = torch.stack(H2_rfm_list, dim=0)  # [nr_class, n_proto, L]
            H2_tensor = H2_tensor.permute(0, 2, 1)       # [nr_class, L, n_proto]
            H2_tensor = H2_tensor.reshape(-1, self.n_proto_per_class)
            H2_rfm = self.linear1(H2_tensor).view(self.args.nr_class, self.L)

            H2_tensor = torch.stack(H2_raw_list, dim=0)  # [nr_class, n_proto, L]
            H2_tensor = H2_tensor.permute(0, 2, 1)       # [nr_class, L, n_proto]
            H2_tensor = H2_tensor.reshape(-1, self.n_proto_per_class)
            H2_raw = self.linear1(H2_tensor).view(self.args.nr_class, self.L)

        # === DUAL-PATH ATTENTION ===
        # Path 1: RFM-transformed features
        A_rfm = self.compute_attention_scores(H_rfm, H2_rfm)
        
        # Path 2: Raw features (no RFM)
        A_raw = self.compute_attention_scores(H_raw, H2_raw)
        
        # Combine: A_final = A_rfm + λ * A_raw
        A_combined = A_rfm + self.attn_lambda * A_raw
        
        # Normalize
        A = F.softmax(A_combined / math.sqrt(self.L), dim=1)
        
        # Bag representation can be formed from raw features, RFM features, or both.
        if self.bag_repr_mode == 'rfm':
            Z_bag = torch.mm(A, H_rfm)
        elif self.bag_repr_mode == 'concat':
            Z_bag = torch.cat([torch.mm(A, H_raw), torch.mm(A, H_rfm)], dim=1)
        else:
            Z_bag = torch.mm(A, H_raw)
        
        # Classification
        Y_logits = self.classifier(Z_bag)
        H_logits = self.instance_classifier(H_rfm)
        H_prob = F.softmax(H_logits, dim=1)

        return Y_logits, A, H_prob, H2_rfm, H_logits

    def update_rfm(self, egop_matrix, momentum=0.9):
        self.rfm_matrix = momentum * self.rfm_matrix + (1 - momentum) * egop_matrix
        self.rfm_sqrt = compute_rfm_sqrt(self.rfm_matrix)

    def full_loss(self, prediction, target, args, H_logits=None):
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        Y_candidate = (target > 0).float()
        prediction_can = prediction * Y_candidate
        prediction_can_sum = prediction_can.sum(dim=1, keepdim=True).clamp(min=1e-10)
        new_prediction = prediction_can / prediction_can_sum
        
        mp_loss = -target * torch.log(prediction + 1e-10)
        sp_loss = -torch.sum(new_prediction * torch.log(new_prediction + 1e-10))
        
        Y_non_candidate = 1.0 - Y_candidate
        prediction_non = prediction * Y_non_candidate
        neg_prediction = (1.0 - prediction_non) * Y_non_candidate + Y_candidate
        neg_prediction = torch.clamp(neg_prediction, min=1e-10)
        in_loss = -Y_non_candidate * torch.log(neg_prediction)
        
        bag_loss = torch.sum(mp_loss) + args.mu * sp_loss + args.gamma * torch.sum(in_loss)

        if H_logits is not None and self.inst_weight > 0:
            target_expanded = target.repeat(H_logits.shape[0], 1)
            target_dist = target_expanded / target_expanded.sum(dim=1, keepdim=True).clamp(min=1e-10)
            H_log_prob = F.log_softmax(H_logits, dim=1)
            instance_loss = -(target_dist * H_log_prob).sum() / H_logits.shape[0]
            total_loss = bag_loss + self.inst_weight * instance_loss
        else:
            total_loss = bag_loss
        
        return new_prediction, total_loss

    def calculate_objective(self, X, Y, Z, args):
        Y = Y.reshape(-1)
        Y_logits, A, H_prob, _, H_logits = self.forward(X, Z)
        Y_prob = F.softmax(Y_logits, dim=1)
        new_prob, loss = self.full_loss(Y_prob, Y, args, H_logits)
        return loss, new_prob, A, H_prob

    def evaluate_objective(self, X, Z):
        Y_logits, _, _, _, _ = self.forward(X, Z)
        Y_prob = F.softmax(Y_logits, dim=1)
        return Y_prob
    
    def regenerate_soft_labels(self, original_s, model_pred, alpha):
        candidate_mask = (original_s > 0).float()
        filtered_pred = model_pred * candidate_mask
        filtered_pred = filtered_pred / (filtered_pred.sum(dim=-1, keepdim=True) + 1e-10)
        new_s = alpha * original_s + (1 - alpha) * filtered_pred
        new_s = new_s / (new_s.sum(dim=-1, keepdim=True) + 1e-10)
        return new_s
