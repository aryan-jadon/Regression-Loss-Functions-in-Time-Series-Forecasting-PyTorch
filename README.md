# Regression Loss Functions Performance Evaluation in Time Series Forecasting using Temporal Fusion Transformers

```
This repository compares the performance of 8 different regression loss functions used 
in Time Series Forecasting using  Temporal Fusion Transformers. Summary of experiment with 
instructions on how to replicate this experiment can be find below.
```

## About Temporal Fusion Transformers
Authors: Bryan Lim, Sercan Arik, Nicolas Loeff and Tomas Pfister

Paper Link: https://arxiv.org/pdf/1912.09363.pdf 

> Abstract - Multi-horizon forecasting problems often contain a complex mix of inputs -- including static (i.e. time-invariant) 
> covariates, known future inputs, and other exogenous time series that are only observed historically -- without any 
> prior information on how they interact with the target. While several deep learning models have been proposed for 
> multi-step prediction, they typically comprise black-box models which do not account for the full range of inputs 
> present in common scenarios. In this paper, we introduce the Temporal Fusion Transformer (TFT) -- a novel 
> attention-based architecture which combines high-performance multi-horizon forecasting with interpretable insights 
> into temporal dynamics. To learn temporal relationships at different scales, the TFT utilizes recurrent layers for 
> local processing and interpretable self-attention layers for learning long-term dependencies. 
> The TFT also uses specialized components for the judicious selection of relevant features and a series of gating layers 
> to suppress unnecessary components, enabling high performance in a wide range of regimes. On a variety of real-world datasets, 
> we demonstrate significant performance improvements over existing benchmarks, and showcase three practical 
> interpretability use-cases of TFT.

# Experiments Summary
Dataset Used for this experiment - https://www.kaggle.com/datasets/utathya/future-volume-prediction


## How To Replicate This Experiment

### Step 1: Install the Requirements

Installing Pytorch Forecasting 

If you are working windows, you need to first install PyTorch with
```bash
pip install torch -f https://download.pytorch.org/whl/torch_stable.html.
```

Otherwise, you can proceed with
```bash
pip install pytorch-forecasting
```

Alternatively, to installl the package via conda:
```bash
conda install pytorch-forecasting pytorch>=1.7 -c pytorch -c conda-forge
```

Installing TorchMetrics

You can install TorchMetrics using pip or conda:

Python Package Index (PyPI)
```bash
pip install torchmetrics
```

Conda
```bash
conda install -c conda-forge torchmetrics
```

### Step 2: Running Experiment Notebooks
```bash
jupyter notebook
```bash

