#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import random
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def adjust_rho(epochs):
    rho_list = [1.0] * epochs
    for ep in range(epochs):
        rho_list[ep] = (epochs - ep) / epochs
    return rho_list


def one_error(outputs, test_target):
    num_class, num_instance = outputs.shape
    temp_outputs = []
    temp_test_target = []

    for i in range(num_instance):
        temp = test_target[:, i]
        if (np.sum(temp) != num_class) and (np.sum(temp) != 0):
            temp_outputs.append(outputs[:, i])
            temp_test_target.append(temp)

    outputs = np.column_stack(temp_outputs)
    test_target = np.column_stack(temp_test_target)
    num_class, num_instance = outputs.shape

    labels = [None] * num_instance
    not_labels = [None] * num_instance
    label_size = np.zeros(num_instance, dtype=int)

    for i in range(num_instance):
        temp = test_target[:, i]
        label_size[i] = np.sum(temp == 1)
        labels[i] = [j + 1 for j in range(num_class) if temp[j] == 1]
        not_labels[i] = [j + 1 for j in range(num_class) if temp[j] == 0]

    oneerr = 0
    for i in range(num_instance):
        indicator = 0
        temp = outputs[:, i]
        max_index = np.argmax(temp)

        if max_index + 1 in labels[i]:
            indicator = 1

        if indicator == 0:
            oneerr += 1

    one_error_value = oneerr / num_instance
    return one_error_value
