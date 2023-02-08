# Multivariate Forecasting Demo Notebooks

This directory contains a number of Jupyter notebooks designed to help you experiment with methods for Multivariate Time Series Forecasting. Multivariate Time Series Forecasting involves predicting the future values of a set of related time series given historical observations. These related time series may have temporal correlation that can be leveraged to collectively enhancy the accuracy of each forecast. Three recent methods that have performormed strongly in the multivariate forecast setting are NBEATS [NBEATS](https://arxiv.org/abs/1905.10437), [NHITS](https://arxiv.org/pdf/2201.12886.pdf) and [DeepAR](https://arxiv.org/abs/1704.04110). We will be exploring each of these methods in this series of demos. 

## Running on Cluster 
The demos in this directory are easily run on the Vector cluster. First, download the provided datasets locally from [Google Drive Link](https://drive.google.com/drive/folders/1X-CgvkQKpatdPPrAYnWaeGmhA-daLJGr?usp=sharing) and extract the electricity dataset folder. Using JupyterHub (or sftp/rsync), upload the electricity dataset to the cluster in the `forecasting-bootcamp/datasets` folder of your home directory. fOpen the notebook you wish to run by navigating to the file in JupyterHub and selecting it. Once the notebook is open select `forecasting` in the *Change Kernel* dropdown under the **Kernel** tab. Finally, select *Restart and Run All* under **Kernel Tab** to run the entire notebook. 

## Demos

### DeepAR
DeepAR, a recently proposed probabilistic forecasting method, is applied to forecast hourly energy consumption accross a set of households. 

### NBEATS
NBEATS, a recently proposed interpretable forecasting method, is applied to forecast hourly energy consumption accross a set of households. 

### NHITS
NHITS, a recently proposed interpretable method for Long Sequence Time Series Forecasting (LSTF), is applied to a synthetic multivariate dataset to do quantile forecasting. 



