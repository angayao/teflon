# Temporal Feature Locality Network (TeFLoN)

## ABSTRACT
Asset return forecasting poses a significant challenge due to complex, dynamic, and stochastic nature
of financial markets. This inherent uncertainty demands models that can robustly learn across
time and regimes. While recurrent networks, convolutional models, and scaled dot-product atten-
tion dominate current approaches, they often overlook feature-dependent temporal locality unique
to financial time-series. Moreover, these techniques lack structural prior essential for modelling such
data. To bridge these gaps, this paper introduce Temporal Feature Locality Network (TeFLoN),
a novel model that incorporates a newly developed Feature Locality Attention (FLA) technique
that dynamically modulates cross-feature interactions by weighting them inversely to the squared
euclidean distance of their latent representations. This prioritizes feature pairs with aligned predic-
tive patterns while suppressing noisy correlations, thereby encoding domain-aware structural priors
critical for financial asset return forecasting. The proposed TeFLoN model is rigorously evaluated
against 14 state-of-the-art baselines spanning deep learning (Deep Multi-Layer Perceptron, Recurrent
Neural Network, Gated Recurrent Unit, Long Short-Term Memory, Transformer), classical machine
learning (Linear Regression, Random Forest, XGBoost, Support Vector Regressor) and statisti-
cal time-series approaches (ARMA, ARIMA, SARIMA, SARIMAX). Experiments on eight global
equity indices (FTSE, GSPC, IXIC, NYA, NSE, N225, KSE, SSE) reveal that TeFLoN achieves a
mean Directional Accuracy (DA) of 83.56% and mean RMSE of 0.0092 outperforming all baseline
models in consideration. A Diebold–Mariano test (p < 0.0001) confirms the statistical significance
of TeFLoN’s performance gains. These results validate TeFLoN’s ability to leverage locality-aware
feature interactions for superior asset return forecasting.

---

## Overview

[View TeFLoN Architecture (PDF)](./resources/teflon.pdf)


[View FLA (PDF)](./resources/fla.pdf)

---

## Status

🚧 **Note:** This repository is under active development. Documentation will be updated soon.


## Citation

> Currently not available. 

