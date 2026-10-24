# Learning to Program Adaptive Non-Local Observables for Machine Learning

Official implementation for the paper: **Learning to Program Adaptive Non-Local Observables for Machine Learning**.

We propose QFWP-ANO, in which a classical hypernetwork programs a data re-uploading variational quantum circuit (VQC) conditioned on each input: its rotation angles, its adaptive non-local observables, or both.

## Abstract
**TL;DR:** We make adaptive non-local observables input-conditioned with a hypernetwork.

Quantum neural networks (QNNs) are typically built from variational quantum circuits (VQCs), which are limited by local measurements. Adaptive non-local observables (ANO) address this by jointly optimizing circuit parameters and multi-qubit measurements. However, existing ANO-based VQCs learn only a single static observable that remains invariant across all inputs. We propose QFWP-ANO, a novel architecture which employs a classical hypernetwork to dynamically program VQC parameters and/or non-local observables conditioned on each input. On multivariate time-series forecasting across four ETT datasets, QFWP-ANO achieves the lowest MSE in 16 of 20 settings and second-lowest in the remaining four, surpassing ANO-based and other strong baselines. On reinforcement learning tasks, QFWP-ANO consistently surpasses ANO-VQCs. Our results establish input-conditioned ANO as an effective approach for enhancing QNNs.

## Usage

Python 3.11.

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run.sh
python tables.py
python -m rl.plot
```

## Repository Layout

```
models/          QFWP-ANO (all variants, VQC, hypernetwork) and RevIN
exp/             Forecasting training and evaluation loop
data_provider/   ETT datasets and data loaders
rl/              A3C agents, training loop, environments and the RL figure
scripts/         train.sh: one variant on one ETT dataset; train_rl.sh: one arm on one environment
dataset/         ETT-small CSVs
```