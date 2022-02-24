# Univariate and Univariate With Exogenous Features Forecasting Demo Notebooks

This directory contains a number of Jupyter notebooks designed to help you experiment with a variety of forecasting problems.

## Dependencies

In testing, we have identified version incompatibilities with Pandas. Please ensure your environment is configured with `pandas==1.3.0`. You can check your version of Pandas using the command `pip show pandas`. If your version of Pandas is not correct, please install the required version using the command `pip install pandas=1.3.0`.

## Running on Google Colab

The demos in this directory have been tested using Google Colab Pro. In order to run a demo, you can connect your Colab account to GitHub. To make use of provided datasets, you will also need to either connect Colab to your personal Google Drive, or directly upload data files to your Colab instance. 

### Step-by-Step Instructions

- Download the provided datasets from this [Google Drive Link](https://drive.google.com/drive/folders/1X-CgvkQKpatdPPrAYnWaeGmhA-daLJGr?usp=sharing).
- Upload these files to your personal Google Drive, noting their location.
- Navigate to [Google Colab](https://colab.research.google.com/)
- From the **File** menu, select **Upload notebook**. You will be prompted to sign in to a Google account.
- Navigate to the **GitHub** tab. You will be prompted to connect Colab with your GitHub account. (**Note** Your GitHub account must be registered with the forecasting-bootcamp repository in order to access it.)
- Ensure that the **Include private repos** checkbox is checked.
- Select the Repository: VectorInstitute/forecasting-bootcamp
- Click on the notebook you would like to open (e.g. `demos/demo1_prophet_neuralprophet_testset.ipynb`)
- Begin execution of the notebook.
- The second code cell mounts your Google Drive. Follow the on-screen prompts.
- Note that in the third code cell, you will need change the file path according to your Google Drive's file structure. Use the **Files** button on the side bar to navigate the file structure. Locate your dataset, right click on it, and select **Copy path**. 
- Continue execution of the notebook. Wherever other files are read or written, be sure to customize file paths according to your Google Drive's file structure.

## Demos

### Demo 1: Baselines, Prophet, and NeuralProphet

Prophet and NeuralProphet applied to exchange rate forecasting. One configuration of Prophet and three configurations of NeuralProphet are considered. Uses a conventional data splitting approach (80% train, 20% test). Applies a collection of four evaluation metrics (MSE, rMSE, MAE, MAPE) in a consistent fashion over all experiments. We also apply two baseline forecasting methods: persistence and mean window forecasting.

### Demo 2: PyTorch Forecasting

N-BEATS and DeepAR applied to exchange rate forecasting. This demo continues the experiment from Demo 1, bringing in two popular deep learning based forecasting methods. The demo provides working examples of how to configure multivariate datasets for univariate, 'global model' forecasting tasks. 

### Demo 3: Rolling Cross-Validation with Prophet for CPI Forecasting

This notebook demonstrates a rolling cross validation experiment using Prophet. The demo was adapted from part of Vector's contribution to the 2022 edition of Canada's Food Price Report. Rolling cross validation can be used to evaluate a forecasting strategy where all available data up to a cutoff point are used to train a model. This is useful in situations where data are scarce or where it is important to train using the most recently available data. During the bootcamp, this approach should only be considered for lightweight models like Prophet due to added training time requirements. 
