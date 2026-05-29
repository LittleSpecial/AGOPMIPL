#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
RFM-MIPL V7: Hybrid Architecture Training Script

Features:
1. Prototype Attention (original, keeps disambiguation working)
2. Bayesian classifier head (uncertainty estimation)
3. KL regularization (prevents overconfidence)
4. LR warmup (stable training across RFM rounds)
5. Alpha warmup (prevents early mode collapse)
"""

from __future__ import print_function
import argparse
import copy
import json
import os
import sys
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import scipy.io as io
import numpy as np
from torch.autograd import Variable
from dataloader import *
from model import MIPML_V12 as MIPML_V7
from utils import *
from rfm import compute_egop, extract_bag_features
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='RFM-MIPL V7 - Hybrid Architecture')

# Basic training arguments
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--L', type=int, default=128)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--index', type=str, default='index')
parser.add_argument('--ds', type=str, default='CRC-MIPL-SBN')
parser.add_argument('--ds_suffix', type=str, default='')
parser.add_argument('--nr_fea', type=int, default=256)
parser.add_argument('--nr_class', type=int, default=7)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--encoder_type', type=str, default='default', choices=['default', 'transformer'],
                    help='instance encoder type after the base feature extractor')
parser.add_argument('--transformer_layers', type=int, default=2,
                    help='number of transformer layers when encoder_type=transformer')
parser.add_argument('--transformer_heads', type=int, default=4,
                    help='number of transformer attention heads when encoder_type=transformer')
parser.add_argument('--transformer_ffn_mult', type=int, default=4,
                    help='FFN width multiplier for the transformer encoder')
parser.add_argument('--transformer_dropout', type=float, default=0.1,
                    help='dropout used by the transformer encoder')

# Loss weights (ELIMIPL style)
parser.add_argument('--mu', type=float, default=0.1, help='sparsity loss weight')
parser.add_argument('--gamma', type=float, default=0.5, help='inhibition loss weight')
parser.add_argument('--inst_weight', type=float, default=0.5, help='instance aux loss weight')
parser.add_argument('--proto_agg', type=str, default='mean', choices=['mean', 'linear'], help='prototype aggregation: mean (V8) or linear (V1)')
parser.add_argument('--attn_lambda', type=float, default=0.3, help='weight for raw attention path in dual-path attention')
parser.add_argument('--bag_repr_mode', type=str, default='raw', choices=['raw', 'rfm', 'concat'],
                    help='which feature space to use for final bag representation')
parser.add_argument('--n_proto_per_class', type=int, default=0,
                    help='number of prototypes per class; 0 means use nr_class')

# RFM arguments
parser.add_argument('--rfm_rounds', type=int, default=3)
parser.add_argument('--epochs_per_round', type=int, default=100)
parser.add_argument('--rfm_momentum', type=float, default=0.0)
parser.add_argument('--no_rfm', action='store_true', default=False)
parser.add_argument('--no_disamb', action='store_true', default=False)

# V7 Bayesian arguments
parser.add_argument('--nr_samples', type=int, default=10, help='Bayesian classifier samples')
parser.add_argument('--kld_weight', type=float, default=0.01, help='KL divergence weight')

# Warmup arguments (NEW)
parser.add_argument('--lr_warmup_epochs', type=int, default=10, help='LR warmup epochs per RFM round')
parser.add_argument('--alpha_warmup_ratio', type=float, default=0.2, help='Alpha warmup ratio (first 20% epochs)')
parser.add_argument('--fold_start', type=int, default=1, help='1-based inclusive fold start')
parser.add_argument('--fold_end', type=int, default=10, help='1-based inclusive fold end')
parser.add_argument('--output_dir', type=str, default='', help='optional output directory for checkpoints and summaries')
parser.add_argument('--exp_name', type=str, default='', help='optional experiment name under output_dir')
parser.add_argument('--save_best_checkpoint', action='store_true', default=False,
                    help='save best checkpoint for each fold')
parser.add_argument('--save_round_checkpoints', action='store_true', default=False,
                    help='save round-end checkpoints for each fold')
parser.add_argument('--save_final_partial_labels', action='store_true', default=False,
                    help='store final train soft labels in the saved checkpoint')

args = parser.parse_args()

seed_everything(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.cuda:
    print('\nGPU is available!')

all_folds = ['index1.mat', 'index2.mat', 'index3.mat', 'index4.mat', 'index5.mat',
             'index6.mat', 'index7.mat', 'index8.mat', 'index9.mat', 'index10.mat']


def sanitize_name(name):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)


def maybe_make_run_dir(args):
    if not args.output_dir:
        return None
    exp_name = args.exp_name
    if not exp_name:
        suffix = args.ds_suffix if args.ds_suffix else 'nosuffix'
        proto = args.n_proto_per_class if args.n_proto_per_class > 0 else 'q'
        exp_name = (
            f"{args.ds}_{suffix}_folds{args.fold_start}-{args.fold_end}_"
            f"r{args.rfm_rounds}e{args.epochs_per_round}_proto{proto}_"
            f"{args.bag_repr_mode}"
        )
    run_dir = os.path.join(args.output_dir, sanitize_name(exp_name))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def serialize_partial_labels(train_set):
    return [
        tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else np.array(tensor)
        for tensor in train_set.train_partial_bag_lab_list
    ]


def save_checkpoint(path, args, model, train_set, fold_idx, best_acc, stage, epoch, train_acc=None, test_acc=None):
    checkpoint = {
        'args': vars(args),
        'model_state_dict': model.state_dict(),
        'prototypes': train_set.prototypes_matrix.detach().cpu(),
        'fold': fold_idx + 1,
        'best_acc': best_acc,
        'stage': stage,
        'epoch': epoch,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }
    if args.save_final_partial_labels:
        checkpoint['train_partial_labels'] = serialize_partial_labels(train_set)
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def adjust_alpha_progressive(epochs, warmup_ratio=0.2):
    """
    Progressive alpha decay with warmup.
    
    First warmup_ratio of epochs: alpha = 1.0 (trust candidate set)
    Remaining: linear decay from 1.0 to 0.0
    """
    warmup_epochs = int(warmup_ratio * epochs)
    decay_epochs = epochs - warmup_epochs
    
    alphas = []
    for ep in range(epochs):
        if ep < warmup_epochs:
            alphas.append(1.0)
        else:
            curr_decay = ep - warmup_epochs
            alpha = 1.0 - (curr_decay / decay_epochs) if decay_epochs > 0 else 0.0
            alphas.append(max(0.0, alpha))
    return alphas


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs, base_lr):
    """
    Create LR scheduler with warmup.
    
    Warmup phase: LR increases from 0 to base_lr
    After warmup: Cosine annealing decay
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        else:
            # Cosine decay after warmup
            progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate(train_loader, loader, model, verbose=True):
    """Model evaluation."""
    model.eval()
    all_true_bag_lab = []
    all_pred_bag_prob = np.empty((0, args.nr_class))
    proto_matrix = train_loader.prototypes_matrix.clone().detach().to(device).to(torch.float32)
    
    with torch.no_grad():
        for data, _, true_bag_lab, _ in loader:
            data = data.to(device).to(torch.float32)
            true_bag_lab_np = true_bag_lab.detach().cpu().numpy().astype(np.float64)
            
            output = model.evaluate_objective(data, proto_matrix)
            output_np = output.detach().cpu().numpy()
            all_pred_bag_prob = np.vstack((all_pred_bag_prob, output_np))
            all_true_bag_lab.append(true_bag_lab_np)
    
    all_true_bag_lab = np.array(all_true_bag_lab)
    pred_labels = np.argmax(all_pred_bag_prob, axis=1)
    true_labels = np.argmax(all_true_bag_lab, axis=1)
    accuracy = np.mean(pred_labels == true_labels)
    
    if verbose:
        print(f'\tAccuracy: {accuracy:.3f}')
    
    return accuracy


def train(train_loader, epoch, alpha_list, current_lr):
    """Training function."""
    model.train()
    train_loss = 0.
    proto_matrix = train_loader.prototypes_matrix.clone().detach().to(device).to(torch.float32)
    
    alpha = alpha_list[epoch - 1] if epoch <= len(alpha_list) else alpha_list[-1]
    
    for data, partial_bag_lab, true_bag_lab, index in train_loader:
        if args.cuda:
            data, partial_bag_lab = data.cuda(), partial_bag_lab.cuda()

        data = Variable(data).to(torch.float32)
        partial_bag_lab = Variable(partial_bag_lab).to(torch.float32)
        
        optimizer.zero_grad()
        
        loss, prediction, attention, H_prob = model.calculate_objective(
            data, partial_bag_lab, proto_matrix, args
        )
        
        train_loss += loss.item()
        
        # Progressive disambiguation
        if not args.no_disamb:
            with torch.no_grad():
                pred = prediction if prediction.dim() == 2 else prediction.unsqueeze(0)
                partial_lab = partial_bag_lab.reshape(1, -1)
                new_label = model.regenerate_soft_labels(partial_lab, pred.detach(), alpha)
                new_label = new_label.squeeze(0).cpu().numpy()
            train_loader.train_partial_bag_lab_list[index] = torch.tensor(new_label)
        
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    
    if epoch == 1 or epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, LR: {current_lr:.6f}, alpha: {alpha:.4f}')
    
    return model, train_loss


import math

if __name__ == "__main__":
    time_s = time.time()
    run_dir = maybe_make_run_dir(args)
    
    num_trial = 1
    num_fold = len(all_folds)
    data_path = os.path.join(args.data_path, args.ds)
    index_path = os.path.join(data_path, args.index)
    
    if args.ds_suffix:
        mat_name = args.ds + '_' + args.ds_suffix + '.mat'
    else:
        mat_name = args.ds + '.mat'
    mat_path = os.path.join(data_path, mat_name)
    
    print(f'\n{"="*60}')
    print(f' RFM-MIPL V7: Hybrid Architecture')
    print(f'{"="*60}')
    print(f'Dataset: {args.ds}')
    print(f'Mat file: {mat_path}')
    print(f'RFM rounds: {args.rfm_rounds}, Epochs/round: {args.epochs_per_round}')
    print(f'\n=== V7 Settings ===')
    print(f'Bayesian samples: {args.nr_samples}')
    print(f'KLD weight: {args.kld_weight}')
    print(f'LR warmup: {args.lr_warmup_epochs} epochs')
    print(f'Alpha warmup: {args.alpha_warmup_ratio*100:.0f}% of epochs')
    print(f'Sparsity (mu): {args.mu}, Inhibition (gamma): {args.gamma}')
    print(f'Bag representation: {args.bag_repr_mode}')
    if args.n_proto_per_class > 0:
        print(f'Prototypes per class: {args.n_proto_per_class}')
    print(f'Fold range: {args.fold_start}-{args.fold_end}')
    if run_dir:
        print(f'Output dir: {run_dir}')
    
    if not os.path.exists(mat_path):
        print(f'\n[ERROR] Mat file not found: {mat_path}')
        sys.exit(1)
    
    # Auto-detect dimensions
    data_mat = io.loadmat(mat_path)
    data = data_mat['data']
    
    for i in range(data.shape[0]):
        if data[i, 0].shape[0] > 0:
            actual_nr_fea = data[i, 0].shape[1]
            if actual_nr_fea != args.nr_fea:
                print(f'[AUTO] Feature dim: {actual_nr_fea}')
                args.nr_fea = actual_nr_fea
            break
    
    for key in ['bag_lab', 'partial_bag_lab']:
        if key in data_mat:
            lab_data = data_mat[key]
            if hasattr(lab_data, 'shape') and len(lab_data.shape) >= 2 and lab_data.shape[1] > 1:
                if lab_data.shape[1] != args.nr_class:
                    print(f'[AUTO] Classes: {lab_data.shape[1]}')
                    args.nr_class = lab_data.shape[1]
                break
    
    all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed = load_data_mat(
        mat_path, args.nr_fea, args.nr_class, normalize=args.normalize)
    
    print(f'\nData loaded: {len(bag_lab)} bags')
    
    all_fold_results = []
    selected_fold_indices = range(max(0, args.fold_start - 1), min(num_fold, args.fold_end))
    if args.n_proto_per_class <= 0:
        args.n_proto_per_class = args.nr_class
    
    for trial_i in range(num_trial):
        for fold_i in selected_fold_indices:
            print(f'\n{"="*50}')
            print(f'Trial {trial_i + 1}, Fold {fold_i + 1}')
            print(f'{"="*50}')
            
            idx_file = index_path + '/' + all_folds[fold_i]
            
            if not os.path.exists(idx_file):
                print(f'[WARN] Index not found: {idx_file}')
                continue
            
            idx_tr, idx_te = load_idx_mat(idx_file)
            print(f'Train: {len(idx_tr)}, Test: {len(idx_te)}')
            
            train_set = MIPMLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, 
                                        partial_bag_lab_processed, idx_tr, idx_te, args.nr_fea, 
                                        train=True, normalize=args.normalize)
            test_set = MIPMLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, 
                                       partial_bag_lab_processed, idx_tr, idx_te, args.nr_fea, 
                                       train=False, normalize=args.normalize)

            train_set.clear_classes(args)
            train_set.classify_by_partial_label(args)
            train_set.generate_prototypes(args, device)
            
            # Initialize V7 model
            model = MIPML_V7(args)
            if args.cuda:
                model.cuda()
            
            print(f'[Model] V7 Hybrid initialized')
            
            best_acc = 0.0
            best_train_acc = None
            best_epoch = None
            best_round = None
            best_state = None
            total_epochs = 0
            fold_time_start = time.time()
            timing = {
                'fold': fold_i + 1,
                'rounds': [],
                'fold_wall_clock_sec': None,
            }
            
            for rfm_round in range(args.rfm_rounds):
                print(f'\n--- RFM Round {rfm_round + 1}/{args.rfm_rounds} ---')
                round_train_start = time.time()
                if args.cuda:
                    torch.cuda.reset_peak_memory_stats()
                
                # Reset optimizer with warmup scheduler for each RFM round
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                                     nesterov=True, weight_decay=args.reg)
                scheduler = get_warmup_scheduler(optimizer, args.lr_warmup_epochs, 
                                                 args.epochs_per_round, args.lr)
                
                alpha_list = adjust_alpha_progressive(args.epochs_per_round, args.alpha_warmup_ratio)
                
                for epoch in range(1, args.epochs_per_round + 1):
                    total_epochs += 1
                    current_lr = optimizer.param_groups[0]['lr']
                    model, loss = train(train_set, epoch, alpha_list, current_lr)
                    scheduler.step()
                    
                    if epoch % 20 == 0 or epoch == args.epochs_per_round:
                        print(f'\n[Eval] Round {rfm_round + 1}, Epoch {epoch}')
                        print('  Train:', end='')
                        train_acc = evaluate(train_set, train_set, model)
                        print('  Test:', end='')
                        test_acc = evaluate(train_set, test_set, model)
                        
                        if test_acc > best_acc:
                            best_acc = test_acc
                            best_train_acc = train_acc
                            best_epoch = epoch
                            best_round = rfm_round + 1
                            best_state = copy.deepcopy(model.state_dict())
                            print(f'  * Best: {best_acc:.3f}')
                            if run_dir and args.save_best_checkpoint:
                                save_checkpoint(
                                    os.path.join(run_dir, f'fold{fold_i + 1}_best.pt'),
                                    args, model, train_set, fold_i, best_acc,
                                    stage=f'round{rfm_round + 1}_best', epoch=epoch,
                                    train_acc=train_acc, test_acc=test_acc,
                                )
                
                # RFM update
                agop_time = 0.0
                agop_peak_mem_mb = None
                if rfm_round < args.rfm_rounds - 1 and not args.no_rfm:
                    print(f'\n[RFM] Computing EGOP...')
                    if args.cuda:
                        torch.cuda.reset_peak_memory_stats()
                    agop_start = time.time()
                    egop = compute_egop(model, extract_bag_features, train_set, device)
                    model.update_rfm(egop, momentum=args.rfm_momentum)
                    agop_time = time.time() - agop_start
                    if args.cuda:
                        agop_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                if run_dir and args.save_round_checkpoints:
                    save_checkpoint(
                        os.path.join(run_dir, f'fold{fold_i + 1}_round{rfm_round + 1}.pt'),
                        args, model, train_set, fold_i, best_acc,
                        stage=f'round{rfm_round + 1}', epoch=args.epochs_per_round,
                        train_acc=None, test_acc=None,
                    )
                timing['rounds'].append({
                    'round': rfm_round + 1,
                    'train_wall_clock_sec': time.time() - round_train_start,
                    'train_peak_mem_mb': (
                        torch.cuda.max_memory_allocated() / (1024 ** 2) if args.cuda else None
                    ),
                    'agop_update_sec': agop_time,
                    'agop_peak_mem_mb': agop_peak_mem_mb,
                })
            
            if best_state is not None:
                model.load_state_dict(best_state)
            # Final results
            print(f'\n{"="*50}')
            print(f'Final Results')
            print(f'{"="*50}')
            print('Train:', end='')
            final_train = evaluate(train_set, train_set, model)
            print('Test:', end='')
            final_test = evaluate(train_set, test_set, model)
            print(f'Best Test: {best_acc:.3f}')
            timing['fold_wall_clock_sec'] = time.time() - fold_time_start
            
            all_fold_results.append({
                'fold': fold_i + 1,
                'final': final_test,
                'best': best_acc,
                'best_train': best_train_acc,
                'best_round': best_round,
                'best_epoch': best_epoch,
                'timing': timing,
            })
            if run_dir and args.save_round_checkpoints:
                with open(os.path.join(run_dir, f'fold{fold_i + 1}_timing.json'), 'w', encoding='utf-8') as f:
                    json.dump(timing, f, indent=2)
            
            torch.cuda.empty_cache()
    
    # Summary
    print(f'\n{"="*60}')
    print(f'EXPERIMENT SUMMARY')
    print(f'{"="*60}')
    if all_fold_results:
        best_accs = [r['best'] for r in all_fold_results]
        print(f'Mean Best Accuracy: {np.mean(best_accs):.3f} ± {np.std(best_accs):.3f}')
        for r in all_fold_results:
            print(f"  Fold {r['fold']}: {r['best']:.3f}")
        if run_dir:
            with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'args': vars(args),
                    'results': all_fold_results,
                    'mean_best': float(np.mean(best_accs)),
                    'std_best': float(np.std(best_accs)),
                    'total_time_sec': time.time() - time_s,
                }, f, indent=2)
    
    print(f'\nTotal time: {time.time() - time_s:.1f}s')
    print('Done.')
