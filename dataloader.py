#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
import random
import torch
import scipy.io as io
import torch.utils.data as data_utils
from sklearn.cluster import KMeans
import math


def to_categorical(y, nr_class):
    """ one-hot encoding """
    y_list = [0] * nr_class
    for i in y:
        y_list[int(i)] = 1
    y_cate = np.array(y_list)

    return y_cate


def load_data_mat(mat_path, nr_fea, nr_class, normalize=False):
    """ load data from .mat file and return instance features, bag index, instance labels, bag labels, partial bag labels """
    data_mat = io.loadmat(mat_path)
    data = data_mat['data']
    all_ins_fea = []
    
    # Auto-detect actual feature dimension from data
    actual_fea_dim = nr_fea
    for i in range(data.shape[0]):
        if data[i, 0].shape[0] > 0:
            actual_fea_dim = data[i, 0].shape[1]
            break
    
    all_ins_fea_tmp = np.empty((0, actual_fea_dim))
    ins_num, bag_idx_of_ins = [], []
    bag_lab = np.empty((0, nr_class))
    dummy_ins_lab = np.empty((0, nr_class))
    partial_dummy_ins_lab = np.empty((0, nr_class))
    partial_bag_lab, partial_bag_lab_processed = np.empty((0, nr_class)), np.empty((0, nr_class))
    bag_cnt = 1
    for i in range(data.shape[0]):
        if data[i, 0].shape[0] == 0:
            continue
        all_ins_fea = np.vstack((all_ins_fea_tmp, data[i, 0]))
        all_ins_fea_tmp = all_ins_fea
        ins_num_tmp = data[i, 0].shape[0]
        ins_num.append(ins_num_tmp)
        bag_idx_of_ins_tmp = [bag_cnt] * ins_num_tmp
        bag_idx_of_ins = bag_idx_of_ins + bag_idx_of_ins_tmp
        bag_cnt += 1
        # the ground-truth labels of bags
        bag_lab_tmp = list(data[i, 2].flatten() - 1)
        # bag_lab = bag_lab + bag_lab_tmp
        bag_lab_tmp = to_categorical(bag_lab_tmp, nr_class)
        bag_lab_tmp = np.expand_dims(bag_lab_tmp, axis=0)
        bag_lab = np.vstack((bag_lab, bag_lab_tmp))
        dummy_ins_lab_tmp = bag_lab_tmp.repeat(ins_num_tmp, axis=0)
        dummy_ins_lab = np.vstack((dummy_ins_lab, dummy_ins_lab_tmp))
        # the partial labels of bags
        partial_bag_lab_tmp = list(data[i, 1].flatten() - 1)
        partial_bag_lab_tmp = to_categorical(partial_bag_lab_tmp, nr_class)
        partial_bag_lab_tmp = np.expand_dims(partial_bag_lab_tmp, axis=0)
        partial_bag_lab = np.vstack((partial_bag_lab, partial_bag_lab_tmp))
        partial_dummy_ins_lab_tmp = partial_bag_lab_tmp.repeat(ins_num_tmp, axis=0)
        partial_dummy_ins_lab = np.vstack((partial_dummy_ins_lab, partial_dummy_ins_lab_tmp))

    bag_idx_of_ins = np.array(bag_idx_of_ins)
    bag_idx_of_ins = np.expand_dims(bag_idx_of_ins, axis=1)
    bag_lab = np.array(bag_lab)
    dummy_ins_lab = np.array(dummy_ins_lab)
    nr_partial_lab_per_bag = np.expand_dims(np.sum(partial_bag_lab, 1), axis=1)
    partial_bag_lab_processed = partial_bag_lab / nr_partial_lab_per_bag

    if normalize:
        data_mean, data_std = np.mean(all_ins_fea, 0), np.std(all_ins_fea, 0)
        # Add small epsilon to prevent division by zero
        all_ins_fea_norm = (all_ins_fea - data_mean) / (data_std + 1e-8)
        all_ins_fea = all_ins_fea_norm
        print(f'[NORMALIZE] Applied z-score normalization to features')

    all_ins_fea = torch.from_numpy(all_ins_fea)
    bag_idx_of_ins = torch.from_numpy(bag_idx_of_ins)
    dummy_ins_lab = torch.from_numpy(dummy_ins_lab)
    bag_lab = torch.from_numpy(bag_lab)
    partial_bag_lab = torch.from_numpy(partial_bag_lab)
    partial_bag_lab_processed = torch.from_numpy(partial_bag_lab_processed)
    return all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed


def load_idx_mat(idx_file):
    """ load the index of training and testing data """
    idx = io.loadmat(idx_file)
    idx_tr_np = idx['trainIndex']
    idx_te_np = idx['testIndex']
    idx_tr = list(np.array(idx_tr_np).flatten())
    idx_te = list(np.array(idx_te_np).flatten())
    random.shuffle(idx_tr)
    random.shuffle(idx_te)
    return idx_tr, idx_te


class MIPMLDataloader(data_utils.Dataset):
    def __init__(self, all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr, idx_te,
                 nr_fea, train=True, normalize=False):
        self.all_ins_fea = all_ins_fea
        self.bag_idx_of_ins = bag_idx_of_ins
        self.dummy_ins_lab = dummy_ins_lab
        self.bag_lab = bag_lab
        self.partial_bag_lab_processed = partial_bag_lab_processed
        self.idx_tr = idx_tr
        self.idx_te = idx_te
        self.train = train
        self.nr_fea = nr_fea
        
        # Check if feature dimension is a perfect square (for 2D reshape)
        sqrt_val = math.sqrt(self.nr_fea)
        self.is_square_fea = (sqrt_val == int(sqrt_val))
        self.nr_fea_sqrt = int(sqrt_val) if self.is_square_fea else int(math.sqrt(self.nr_fea) + 1)
        
        self.normalize = normalize
        self.classes = {}
        self.total_amount = 0
        self.prototypes_matrix = []
        self.appearance_freq = []

        if self.train:
            self.train_bags_list, self.train_ins_lab_list, self.train_partial_bag_lab_list, \
            self.train_true_bag_lab_list = self._create_bags()
        else:
            self.test_bags_list, self.test_ins_lab_list, self.test_partial_bag_lab_list, \
            self.test_true_bag_lab_list = self._create_bags()

    def _create_bags(self):
        bags_list, ins_lab_list, partial_bag_lab_list, true_bag_lab_list = [], [], [], []
        if self.train:
            for i in self.idx_tr:
                bag_idx_of_ins_a_bag = self.bag_idx_of_ins == i
                bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag)
                bag = self.all_ins_fea[bag_idx_of_ins_a_bag, :]
                ins_lab = self.dummy_ins_lab[bag_idx_of_ins_a_bag]
                partial_bag_lab = self.partial_bag_lab_processed[i - 1, :]
                partial_bag_lab = np.expand_dims(partial_bag_lab, axis=0)
                partial_bag_lab = torch.tensor(partial_bag_lab)
                true_bag_lab = self.bag_lab[i - 1]
                # Reshape based on feature dimension type
                if self.is_square_fea:
                    # 2D reshape for perfect square features (e.g., 16x16)
                    bag = bag.reshape(bag.shape[0], 1, self.nr_fea_sqrt, self.nr_fea_sqrt)
                else:
                    # Keep as 1D for non-square features (e.g., 15-dim)
                    bag = bag.reshape(bag.shape[0], self.nr_fea)
                bags_list.append(bag)
                ins_lab_list.append(ins_lab)
                partial_bag_lab_list.append(partial_bag_lab)
                true_bag_lab_list.append(true_bag_lab)
        else:
            for i in self.idx_te:
                bag_idx_of_ins_a_bag = self.bag_idx_of_ins == i
                bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag)
                bag = self.all_ins_fea[bag_idx_of_ins_a_bag, :]
                ins_lab = self.dummy_ins_lab[bag_idx_of_ins_a_bag]
                partial_bag_lab = self.partial_bag_lab_processed[i - 1, :]
                partial_bag_lab = np.expand_dims(partial_bag_lab, axis=0)
                partial_bag_lab = torch.tensor(partial_bag_lab)
                true_bag_lab = self.bag_lab[i - 1]
                # Reshape based on feature dimension type
                if self.is_square_fea:
                    bag = bag.reshape(bag.shape[0], 1, self.nr_fea_sqrt, self.nr_fea_sqrt)
                else:
                    bag = bag.reshape(bag.shape[0], self.nr_fea)
                bags_list.append(bag)
                ins_lab_list.append(ins_lab)
                partial_bag_lab_list.append(partial_bag_lab)
                true_bag_lab_list.append(true_bag_lab)

        return bags_list, ins_lab_list, partial_bag_lab_list, true_bag_lab_list

    def __len__(self):
        if self.train:
            return len(self.train_ins_lab_list)
        else:
            return len(self.test_ins_lab_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            partial_bag_label = self.train_partial_bag_lab_list[index]
            true_bag_label = self.train_true_bag_lab_list[index]
        else:
            bag = self.test_bags_list[index]
            partial_bag_label = self.test_partial_bag_lab_list[index]
            true_bag_label = self.test_true_bag_lab_list[index]

        return bag, partial_bag_label, true_bag_label, index

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.train:
            if self.iter_index < len(self.train_ins_lab_list):
                result1, result2, result3, result4 = self.__getitem__(self.iter_index)
                self.iter_index += 1
                return result1, result2, result3, result4
            else:
                raise StopIteration
        else:
            if self.iter_index < len(self.test_ins_lab_list):
                result1, result2, result3, result4 = self.__getitem__(self.iter_index)
                self.iter_index += 1
                return result1, result2, result3, result4
            else:
                raise StopIteration

    def classify_by_partial_label(self, args):
        for i in range(1, args.nr_class + 1):
            self.classes[i] = []

        for i in range(len(self.train_bags_list)):
            self.total_amount += self.train_bags_list[i].shape[0]
            bag = self.train_bags_list[i]
            bag = bag.view(bag.shape[0], -1)
            partial_bag_label = self.partial_bag_lab_processed[i]
            for j in range(len(partial_bag_label)):
                if partial_bag_label[j] != 0:
                    for _, item in enumerate(bag):
                        self.classes[j + 1].append(item)

        for key in self.classes:
            if len(self.classes[key]) > 0:
                self.classes[key] = torch.stack(self.classes[key])
            else:
                self.classes[key] = torch.tensor([])

        for i in range(0, args.nr_class):
            self.appearance_freq.append(0.5)

    def clear_classes(self, args):
        self.classes = {}
        self.total_amount = 0
        for i in range(1, args.nr_class + 1):
            self.classes[i] = []

    def generate_prototypes(self, args, device):
        n_proto = getattr(args, 'n_proto_per_class', args.nr_class)
        for i in range(1, args.nr_class + 1):
            if len(self.classes[i]) == 0:
                # Create zero-filled placeholder for empty class
                placeholder = torch.zeros((n_proto, self.nr_fea), device=device, dtype=torch.float32)
                self.prototypes_matrix.append(placeholder)
                continue
            one_class = np.array(self.classes[i])
            kmeans = KMeans(n_clusters=n_proto, random_state=0).fit(one_class)
            prototypes = torch.tensor(kmeans.cluster_centers_, device=device).to(torch.float32)
            self.prototypes_matrix.append(prototypes)

        self.prototypes_matrix = torch.stack(self.prototypes_matrix)
        # Reshape based on feature dimension type
        if self.is_square_fea:
            self.prototypes_matrix = self.prototypes_matrix.view(
                args.nr_class, n_proto, self.nr_fea_sqrt, self.nr_fea_sqrt
            )
        else:
            # Keep as 1D for non-square features
            self.prototypes_matrix = self.prototypes_matrix.view(args.nr_class, n_proto, self.nr_fea)
