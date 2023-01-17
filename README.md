# Regression Loss Functions Performance Evaluation in Time Series Forecasting using Temporal Fusion Transformers
[![DOI](https://zenodo.org/badge/539733786.svg)](https://zenodo.org/badge/latestdoi/539733786)

```
This repository compares the performance of 8 different regression loss functions used 
in Time Series Forecasting using  Temporal Fusion Transformers. Summary of experiment with 
instructions on how to replicate this experiment can be find below.
```

## About Temporal Fusion Transformers

Paper Link: https://arxiv.org/pdf/1912.09363.pdf 

Authors: Bryan Lim, Sercan Arik, Nicolas Loeff and Tomas Pfister

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

## Experiments Summary and Our Paper

### Cite Our Paper

```
@misc{jadon2022comprehensive,
      title={A Comprehensive Survey of Regression Based Loss Functions for Time Series Forecasting}, 
      author={Aryan Jadon and Avinash Patil and Shruti Jadon},
      year={2022},
      eprint={2211.02989},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Our Paper Link - https://arxiv.org/abs/2211.02989

![Summary of Loss Functions](https://github.com/aryan-jadon/Regression-Loss-Functions-in-Time-Series-Forecasting-Tensorflow/blob/main/loss_functions_plots/Loss-Functions-Summary.png)

### Dataset Used for this experiment - https://www.kaggle.com/datasets/utathya/future-volume-prediction

Parameters Used During Experiments -

| Parameters                                       | Value |
|:------------------------------------------------------|:-----:| 
| learning_rate                                         | 0.03  |
| hidden_size                                           |  16   |
| attention_head_size                                   |   1   |
| dropout                                               |  0.1  |
| hidden_continuous_size                                |  8    |


| Loss Function                                  |    Value    |
|:-----------------------------------------------|:-----------:| 
| Mean Absolute Percentage Error Loss            | 258170080.0 |
| Mean Squared Error Loss                        |  501227.06  |
| Quantile Loss                                  |  1462.5258  |
| Root Mean Square Error Loss                    |   683.93    |
| Mean Absolute Scaled Error Loss                |   288.15    |
| Mean Absolute Error Loss                       |   264.46    |
| Symmetric Mean Absolute Percentage Loss        |    0.40     |
| Mean Squared Log Error Loss                    |   0.34      |


## Replicate This Experiment

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

Alternatively, to install the package via conda:
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
```

