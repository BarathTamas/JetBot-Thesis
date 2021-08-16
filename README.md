# JetBot Thesis

## Introduction
The Thesis consists of two parts. In the first part a customized JetBot with a Jetson Nano 4GB board is used to perform measurements around a room. In the second part unsupervised time series feature selection is used to identify the _k_ sensor locations with the "maximal information".

## Data Collection
### Road following demo
Video of road following with stop/move classification from the POV of the JetBot. Video is at ~3x real speed. At the sensor stops (green rectangles) the robot performs measurements for 10 second without processing images, hence the short "cuts".
In every frame, the JetBot has to 1. estimate where to go based on the marking tape (prediction marked with a red circle), 2. decide whether a stop has been reached. The robot is using a multi-tasking CNN (with EfficientNet b0 as the encoder), jointly retrained for the regression and classification tasks.

https://user-images.githubusercontent.com/44137494/129424808-45d1cc8e-4523-45ef-bdb2-94560483d9dd.mp4

## Sensor selection
### Concrete Autoencoder
#### Source used
The concrete autoencoder is an end-to-end differentiable method for global feature selection, which efficiently identifies a subset of the most informative features and simultaneously learns a neural network to reconstruct the input data from the selected features. The method can be applied to unsupervised and supervised settings, and is a modification of the standard autoencoder.

For more details, see the author's paper: ["Concrete Autoencoders for Differentiable Feature Selection and Reconstruction"](https://arxiv.org/abs/1901.09346), *ICML 2019*, and please use the citation below.

```
@article{abid2019concrete,
  title={Concrete Autoencoders for Differentiable Feature Selection and Reconstruction},
  author={Abid, Abubakar and Balin, Muhammed Fatih and Zou, James},
  journal={arXiv preprint arXiv:1901.09346},
  year={2019}
}
```
#### Added modifications:
- RNN/LSTM compatibility
- IPython display outputs during training
- custom checkpoint that saves the best _converged_ model
### Hilbert-Schmidt Independence Criterion (HSIC) and Centered Kernel Alignment (CKA)
#### Introduction
The goal of HSIC and CKA is to establish dependence between two series of arbitrary dimensions. An analogy could be HSIC being a non-linear "covariance", and CKA the normalized version, "correlation". However, HSIC and CKA do not require matching dimensions for the compared series, which makes them more similar to a non-linear canonical correlation analysis (CCA).
#### Contribution
My implementation is more efficient than the others I could find on GitHub. In addition I also included code for the incomplete Cholesky approximation, but the latter is just Python adaptation of John Shawe-Taylor and Nello Cristianini's Matlab code (Kernel methods for pattern analysis, 5.2 Computing projections: page 126–129.).
#### Features
- CKA
- HSIC with test statistic
- incomplete Cholesky decomposition
