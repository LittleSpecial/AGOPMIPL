# Enhanced Multi-Instance Partial Label Learning via Average Gradient Outer Product

<p align="center">
  <a href="https://openreview.net/"><img src="https://img.shields.io/badge/OpenReview-AGOPMIPL-blue"></a>
  <a href="https://proceedings.mlr.press/"><img src="https://img.shields.io/badge/PMLR-2026-darkred"></a>
  <img src="https://img.shields.io/badge/ICML-2026-success">
</p>

Official implementation of **AGOPMIPL** — "Enhanced Multi-Instance Partial Label Learning via Average Gradient Outer Product" (ICML 2026).

Nan Cao\*, Xu Zhao\*, Teng Zhang &nbsp;(\*equal contribution) — Huazhong University of Science and Technology

> **TL;DR.** Under the dual weak supervision of multi-instance *and* partial-label learning, we use the Average Gradient Outer Product (AGOP) to reshape the feature metric *before* attention, so the model locates the true key instances instead of being misled by false-positive candidate labels.

## Abstract

Multi-instance partial-label learning (MIPL) annotates each training bag with a candidate label set containing one true label and several false positives. MIPL is strictly harder than the sum of multi-instance and partial-label learning: when both signals are weak, even the bag-level supervision is unreliable, making the association between key instances and the true label especially hard to recover. We propose AGOPMIPL, which uses the average gradient outer product to amplify discriminative feature directions and improve key-instance identification; feature prototypes and a progressive disambiguation strategy further suppress noisy candidates. A key advantage is that AGOP mitigates the weak supervision in both spaces with a single mechanism, rather than requiring separate modules. On four MIPL benchmarks and the real-world CRC-MIPL dataset, AGOPMIPL consistently outperforms five state-of-the-art baselines, with up to 25.9% relative gain on CRC-MIPL-KMeansSeg.

## Method

<p align="center">
  <img src="assets/architecture.png" width="90%">
</p>

The framework has three parts. (1) Instance prototype construction runs k-means per class to obtain class prototypes. (2) The AGOP feature-learning module computes the average gradient outer product `G_t` from the bag-level classifier and transforms both instance features `H` and prototype features `E` by `M_t^{1/2}`, amplifying discriminative directions. (3) A dual-path attention aggregates instances into the bag representation using scores from both the raw and the AGOP-transformed features, and a progressive label-disambiguation step refines the soft target during training.

## Setup

```bash
conda create -n agopmipl python=3.10 -y
conda activate agopmipl
pip install -r requirements.txt
# install a CUDA build of PyTorch matching your machine, e.g.
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Datasets

We use the four standard MIPL benchmarks (MNIST-MIPL, FMNIST-MIPL, Birdsong-MIPL, SIVAL-MIPL) and the real-world CRC-MIPL dataset, in the same `.mat` format as DEMIPL / ELIMIPL. Place them under `./data` as:

```
data/
├── MNIST_MIPL/
│   ├── MNIST_MIPL_r1.mat   MNIST_MIPL_r2.mat   MNIST_MIPL_r3.mat
│   └── index/index1.mat ... index10.mat
├── FMNIST_MIPL/   (same layout)
├── Birdsong_MIPL/ (same layout)
├── SIVAL_MIPL/    (same layout)
└── CRC-MIPL-KMeansSeg/
    ├── CRC-MIPL-KMeansSeg.mat
    └── index/index1.mat ... index10.mat
```

`_r{1,2,3}` denotes the number of false-positive candidate labels (candidate-set noise level); `index{k}.mat` is the k-th train/test split.

## Usage

Train and evaluate with `main.py`, pointing `--ds` at the dataset directory under `./data`. For example, FMNIST-MIPL at candidate-set noise level `r = 1`, averaged over the 10 benchmark splits:

```bash
python main.py \
  --ds FMNIST_MIPL --ds_suffix r1 --data_path ./data \
  --nr_fea 784 --nr_class 5 \
  --rfm_rounds 4 --epochs_per_round 100 \
  --lr 0.005 --reg 1e-4 --normalize \
  --inst_weight 0.0 --proto_agg linear --n_proto_per_class 5 \
  --attn_lambda 1.0 --bag_repr_mode raw \
  --fold_start 1 --fold_end 10
```

Recommended per-dataset hyperparameters:

| Dataset | nr_fea | nr_class | rfm_rounds | epochs/round | reg | inst_weight | proto_agg | n_proto/class |
|---|---|---|---|---|---|---|---|---|
| MNIST-MIPL  | 784 | 5  | 4 | 100 | 1e-4 | 0.0 | linear | 5  |
| FMNIST-MIPL | 784 | 5  | 4 | 100 | 1e-4 | 0.0 | linear | 5  |
| SIVAL-MIPL  | 30  | 25 | 3 | 80  | 1e-3 | 0.1 | mean   | 25 |
| Birdsong-MIPL | 38 | 13 | 8 | 50 | 1e-3 | 0.1 | mean | 13 |
| CRC-MIPL-KMeansSeg | 6 | 7 | 3 | 100 | 1e-4 | 0.0 | linear | 7 |

Shared across datasets: `--lr 0.005 --normalize --attn_lambda 1.0 --bag_repr_mode raw`. `--ds_suffix r{1,2,3}` selects the candidate-set noise level (number of false positives); CRC-MIPL uses fixed candidate sets and takes no suffix. The other CRC-MIPL embeddings differ only in `--ds` and `--nr_fea` (SBN: 15, SIFT: 128, Row: 15). Run `python main.py --help` for the full argument list.

## Citation

```bibtex
@inproceedings{cao2026agopmipl,
  title     = {Enhanced Multi-Instance Partial Label Learning via Average Gradient Outer Product},
  author    = {Cao, Nan and Zhao, Xu and Zhang, Teng},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR}
}
```

## Acknowledgement

This work builds on the Recursive Feature Machine / AGOP framework of Radhakrishnan et al. (*Science*, 2024). The MIPL benchmark data format follows DEMIPL and ELIMIPL.

## TODO

- [ ] Replace the OpenReview and PMLR badge links with the final paper URLs once available
- [ ] Add an arXiv preprint link
- [ ] Add ICML 2026 slides / poster
- [ ] Update the BibTeX with the final volume and page numbers after the PMLR proceedings are published
- [ ] Expand dataset download/preparation instructions (the .mat benchmark files)
- [ ] Release pretrained checkpoints (optional)
