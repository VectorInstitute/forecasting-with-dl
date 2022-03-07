# Multivariate Long Sequence Time Series Forecasting Demo Notebooks

This directory contains a number of Jupyter notebooks designed to help you experiment with methods for Multivariate Long Sequence Time Series forecasting (LSTF). LSTF is a branch of time series forecating that is characterized by long input and output sequences. Being able to model the the behaviour of a system over long time horizons is essential to plan effectively in a variety of domains including economics, electricity consumption and transportation. Because of the large input and output space, standard architectures like RNN and Transformers do not scale well to LSTF. Recently several methods for LSTF have been proposed of which we will be exploring the most performant: [Autoformer](https://arxiv.org/abs/2106.13008) and [NHITS](https://arxiv.org/abs/2201.12886). 

## Running on Cluster 
The demos in this directory are all tested on the cluster. Open the notebook you wish to run by navigating to the file in JupyterHub and selecting it. Once the notebook is open select `forecasting` in the *Change Kernel* dropdown under the **Kernel** tab. Finally, select *Restart and Run All* under **Kernel Tab** to run the entire notebook. The notebooks in this directory will automatically download the dataset in the same directy as the notebook so each demo will out of the box without the need to download or alter filepaths to the dataset.  

## Demos

### Autoformer
Autoformer, a recently proposed transformer-based method for LSTF, is applied to forecast hourly road occupancy data from 862 sensors from San Francisco Bay Area freeways. First, the demo outlines how to load the data and split it into train, validation and test sets. An autoformer model is subsequently initialized with an appropriate set of hyperparmeters given the dataset. The model is trained using the training data and evaluated at regular intervals using the validation data. Once the model is finished training, the model is evaluated on the test set and the predicitons are visualized. 

### NHITS
NHITS, a recently proposed LSTF model based on [NBEATS](https://arxiv.org/abs/1905.10437), is applied to forecast hourly road occupancy data from 862 sensors from San Francisco Bay Area freeways. First, the demo outlines how to load the data and split it into train, validation and test sets. An autoformer model is subsequently initialized with an appropriate set of hyperparmeters given the dataset. The model is trained using the training data and evaluated at regular intervals using the validation data. Once the model is finished training, the model is evaluated on the test set and the predicitons are visualized. 




