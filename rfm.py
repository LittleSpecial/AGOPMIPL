#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
RFM (Recursive Feature Machine) Module

Implementation following the official Deep Neural Feature Ansatz repository:
https://github.com/aradha/deep_neural_feature_ansatz

Key components:
1. get_jacobian() - Compute Jacobian using functorch
2. compute_egop() - Compute Expected Gradient Outer Product: G = E[J^T @ J]
3. compute_rfm_sqrt() - Compute matrix square root for feature transformation
"""

import torch
import torch.nn as nn
import numpy as np

# Try to import functorch, fall back to torch.func for newer PyTorch versions
try:
    from functorch import jacrev, vmap
except ImportError:
    from torch.func import jacrev, vmap


def get_jacobian(model, data):
    """
    Compute the Jacobian of the model output with respect to input.
    
    Args:
        model: Neural network model (should be a function from input to output)
        data: Input tensor [batch_size, feature_dim]
    
    Returns:
        Jacobian tensor [batch_size, num_classes, feature_dim]
    """
    with torch.no_grad():
        # vmap applies jacrev across the batch dimension
        # jacrev computes the Jacobian of the model
        J = vmap(jacrev(model))(data)
        # J shape: [batch_size, num_classes, feature_dim]
        return J


def compute_egop(model, feature_extractor, dataloader, device, centering=False):
    """
    Compute Expected Gradient Outer Product (EGOP) matrix.
    
    EGOP captures which input features are most important for the classifier's
    predictions by averaging the outer product of Jacobians.
    
    G = (1/n) * Σ J_i^T @ J_i
    
    where J_i is the Jacobian of the classifier output w.r.t. the features
    for sample i.
    
    Args:
        model: The full MIPML model
        feature_extractor: Function to extract features from the model
        dataloader: Training data loader
        device: torch device
        centering: Whether to center the Jacobians (subtract mean)
    
    Returns:
        EGOP matrix [feature_dim, feature_dim]
    """
    model.eval()
    
    # Get the classifier layer (last layer that maps features to classes)
    classifier = model.classifier
    
    # Collect all features from the training set
    all_features = []
    
    print("[RFM] Extracting features for EGOP computation...")
    
    with torch.no_grad():
        for data, partial_bag_lab, true_bag_lab, index in dataloader:
            data = data.to(device).to(torch.float32)
            
            # Extract features using the model's feature extractor
            # This should give us the bag-level representation before classifier
            features = feature_extractor(model, data, dataloader.prototypes_matrix.to(device))
            all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    n_samples = all_features.shape[0]
    feature_dim = all_features.shape[1]
    
    # In concat mode the classifier sees [raw ; rfm], but the recursive metric
    # update should only act on the transformed half that is actually remapped
    # by rfm_sqrt. Otherwise the EGOP becomes 2L x 2L while the feature metric
    # remains defined on the L-dimensional transformed space.
    use_concat_rfm_half = getattr(model, 'bag_repr_mode', 'raw') == 'concat' and hasattr(model, 'L')
    effective_feature_dim = feature_dim
    if use_concat_rfm_half:
        effective_feature_dim = model.L
        print(
            f"[RFM] Concat mode detected; using the transformed half of the classifier "
            f"for EGOP (feature_dim={effective_feature_dim})"
        )

    print(f"[RFM] Computing Jacobian for {n_samples} samples, feature_dim={effective_feature_dim}")
    
    # Define classifier as a standalone function for Jacobian computation
    def classifier_fn(x):
        # x: [feature_dim]
        # Need to handle the classifier being an nn.Sequential
        with torch.no_grad():
            # Get the linear layer weights and bias
            for layer in classifier:
                if isinstance(layer, nn.Linear):
                    return x @ layer.weight.T + layer.bias
        return classifier(x.unsqueeze(0)).squeeze(0)
    
    # Compute Jacobians in batches to save memory
    batch_size = 500
    batches = torch.split(all_features, batch_size)
    
    all_jacobians = []
    for batch_idx, batch in enumerate(batches):
        batch = batch.to(device)
        print(f"[RFM] Computing Jacobian for batch {batch_idx + 1}/{len(batches)}")
        
        # For a linear classifier y = Wx + b, the Jacobian is just W
        # So we can compute it more efficiently
        # J[i, j] = ∂y_i / ∂x_j = W[i, j]
        
        # V8 compatibility: classifier might be nn.Linear or nn.Sequential
        if isinstance(classifier, nn.Linear):
            W = classifier.weight.data  # [num_classes, feature_dim]
            if use_concat_rfm_half:
                W = W[:, -model.L:]
            J = W.unsqueeze(0).expand(batch.shape[0], -1, -1)
            all_jacobians.append(J.cpu())
        else:
            # Original model: classifier is Sequential
            for layer in classifier:
                if isinstance(layer, nn.Linear):
                    W = layer.weight.data  # [num_classes, feature_dim]
                    if use_concat_rfm_half:
                        W = W[:, -model.L:]
                    J = W.unsqueeze(0).expand(batch.shape[0], -1, -1)
                    all_jacobians.append(J.cpu())
                    break
    
    all_jacobians = torch.cat(all_jacobians, dim=0)  # [n_samples, num_classes, feature_dim]
    
    if centering:
        J_mean = torch.mean(all_jacobians, dim=0, keepdim=True)
        all_jacobians = all_jacobians - J_mean
    
    # Compute EGOP: G = (1/n) * Σ J_i^T @ J_i
    # For each sample i: J_i is [num_classes, feature_dim]
    # J_i^T @ J_i is [feature_dim, feature_dim]
    print("[RFM] Computing EGOP matrix...")
    
    # Use einsum for efficient computation
    # all_jacobians: [n, c, d] where n=samples, c=classes, d=features
    # We want: G = (1/n) * Σ_i (J_i^T @ J_i) = (1/n) * Σ_i Σ_c J_i[c,:].T @ J_i[c,:]
    # This is equivalent to: G[d1, d2] = (1/n) * Σ_i Σ_c J_i[c, d1] * J_i[c, d2]
    
    J = all_jacobians  # [n, c, d]
    G = torch.einsum('ncd,ncD->dD', J, J) / n_samples  # [d, d]
    
    print(f"[RFM] EGOP matrix computed, shape={G.shape}, norm={torch.norm(G).item():.4f}")
    
    model.train()
    return G.to(device)


def compute_rfm_sqrt(M, eps=1e-6):
    """
    Compute the matrix square root of the RFM matrix M.
    
    Uses eigenvalue decomposition for numerical stability:
    M = V @ diag(λ) @ V^T
    M^(1/2) = V @ diag(√λ) @ V^T
    
    Args:
        M: Symmetric positive semi-definite matrix [d, d]
        eps: Small value to ensure positivity
    
    Returns:
        M^(1/2): Matrix square root [d, d]
    """
    # Ensure M is symmetric
    M = (M + M.T) / 2
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    
    # Debug: print eigenvalue statistics
    print(f"[RFM] Eigenvalues - min: {eigenvalues.min().item():.6f}, max: {eigenvalues.max().item():.6f}, mean: {eigenvalues.mean().item():.6f}")
    
    # Clamp eigenvalues to be positive
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    
    # Compute square root
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    
    # Reconstruct M^(1/2)
    M_sqrt = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    
    print(f"[RFM] M_sqrt computed, norm: {torch.norm(M_sqrt).item():.4f}")
    
    return M_sqrt


def update_rfm_matrix(current_M, new_egop, momentum=0.9):
    """
    Update RFM matrix using exponential moving average.
    
    M_new = momentum * M_current + (1 - momentum) * EGOP
    
    Args:
        current_M: Current RFM matrix [d, d]
        new_egop: Newly computed EGOP matrix [d, d]
        momentum: EMA momentum (default 0.9)
    
    Returns:
        Updated RFM matrix [d, d]
    """
    return momentum * current_M + (1 - momentum) * new_egop


def extract_bag_features(model, data, proto_matrix):
    """
    Extract bag-level features from the model (before classifier).
    
    This function replicates the forward pass up to the attention aggregation
    to get the bag representation Z.
    
    Args:
        model: MIPML model (V1, V5, or V6)
        data: Input bag [1, n_instances, 1, H, W] or [1, n_instances, nr_fea]
        proto_matrix: Prototype matrix for each class
    
    Returns:
        Bag representation Z [1, feature_dim]
    """
    import torch.nn.functional as F
    import math
    
    # For V6 model, use its built-in extract_features method
    if hasattr(model, 'extract_features'):
        X = data.squeeze(0) if data.dim() > 3 else data
        H = model.extract_features(X if X.dim() >= 2 else X.unsqueeze(0))
        # Simple attention-weighted aggregation for EGOP computation
        A = F.softmax(torch.ones(1, H.shape[0], device=H.device), dim=1)
        Z = torch.mm(A, H)
        return Z
    
    # Original implementation for V1/V5/V12 models
    # Feature extraction (Equation 1)
    X = data.squeeze(0)
    
    if hasattr(model, 'extract_instance_embeddings'):
        H_raw = model.extract_instance_embeddings(X)
    elif model.use_cnn:
        H_raw = model.feature_extractor_part1(X)
        H_raw = H_raw.view(-1, model.cnn_output_size)
        H_raw = model.feature_extractor_part2(H_raw)
    else:
        H_raw = X.view(X.shape[0], -1)
        H_raw = model.feature_extractor_part2(H_raw)
    
    # Apply RFM transformation if available
    if hasattr(model, 'rfm_sqrt') and model.rfm_sqrt is not None:
        H_rfm = H_raw @ model.rfm_sqrt
    else:
        H_rfm = H_raw
    
    # Prototype feature extraction (Equation 2)
    if hasattr(model, 'extract_instance_embeddings'):
        Z = proto_matrix
        H2_list = [model.extract_instance_embeddings(Z[i]) for i in range(len(Z))]
    elif model.use_cnn:
        Z = proto_matrix
        Z_list = [Z[i].squeeze(1) for i in range(len(Z))]
        nr_fea_sqrt = model.nr_fea_sqrt
        Z_list = [Z_list[i].view(-1, 1, nr_fea_sqrt, nr_fea_sqrt) for i in range(len(Z_list))]
        H2_list = [model.feature_extractor_part1(Z_list[i]) for i in range(len(Z))]
        H2_list = [H2_list[i].view(-1, model.cnn_output_size) for i in range(len(H2_list))]
        H2_list = [model.feature_extractor_part2(H2_list[i]) for i in range(len(H2_list))]
    else:
        Z = proto_matrix
        Z_list = [Z[i].view(Z[i].shape[0], -1) for i in range(len(Z))]  # Flatten
        H2_list = [model.feature_extractor_part2(Z_list[i]) for i in range(len(Z))]
    
    H2_rfm_list = [h2 @ model.rfm_sqrt if hasattr(model, 'rfm_sqrt') and model.rfm_sqrt is not None else h2 for h2 in H2_list]
    
    # Prototype aggregation (Equation 3)
    # V8 compatibility: use mean pooling if linear1 doesn't exist
    if hasattr(model, 'linear1'):
        H2_tensor = torch.stack(H2_rfm_list, dim=0)
        H2_tensor = H2_tensor.permute(0, 2, 1)
        H2_tensor = H2_tensor.reshape(-1, model.n_proto_per_class)
        H2 = model.linear1(H2_tensor)
        H2 = H2.view(model.args.nr_class, model.args.L)
    else:
        # V8: Use mean pooling instead
        H2_centers = [torch.mean(h2, dim=0) for h2 in H2_rfm_list]
        H2 = torch.stack(H2_centers, dim=0)  # [nr_class, L]
    
    # Attention (Equations 4-6)
    A_V = model.att_layer_V(H_rfm, model.linear_V.weight, model.linear_V.bias)
    A_U = model.att_layer_U(H2, model.linear_U.weight, model.linear_U.bias)
    A = model.attention_weights(A_V * A_U.T)
    A = torch.transpose(A, 1, 0)
    A = A / math.sqrt(model.args.L)
    A = F.softmax(A, dim=1)
    
    # Match the model's bag-level representation mode.
    bag_repr_mode = getattr(model, 'bag_repr_mode', 'raw')
    if bag_repr_mode == 'rfm':
        Z = torch.mm(A, H_rfm)
    elif bag_repr_mode == 'concat':
        Z = torch.cat([torch.mm(A, H_raw), torch.mm(A, H_rfm)], dim=1)
    else:
        Z = torch.mm(A, H_raw)
    
    return Z
