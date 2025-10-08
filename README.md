# Improved Tacrolimus AUC Prediction UsingNeural Ordinary Differential Equations

Code for the paper:
> Benjamin Maurel, . "Improved Tacrolimus AUC Prediction UsingNeural Ordinary Differential Equations" (2025)
[[arxiv]]

<p align="center">
<img align="middle" src="./assets/viz.gif" width="800" />
</p>

## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Experiments on different datasets

By default, the dataset are downloadeded and processed when script is run for the first time. 

Raw datasets: 
[[MuJoCo]](http://www.cs.toronto.edu/~rtqichen/datasets/HopperPhysics/training.pt)
[[Physionet]](https://physionet.org/physiobank/database/challenge/2012/)
[[Human Activity]](https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity/)

To generate MuJoCo trajectories from scratch, [DeepMind Control Suite](https://github.com/deepmind/dm_control/) is required


### Running different models

### Making the visualization