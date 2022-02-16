# Multivariate Forecasting Demo Notebooks

This directory contains a number of Jupyter notebooks designed to help you experiment with a variety of forecasting problems.

## Dependencies

In testing, we have identified version incompatibilities with Pandas. Please ensure your environment is configured with `pandas==1.3.0`. You can check your version of Pandas using the command `pip show pandas`. If your version of Pandas is not correct, please install the required version using the command `pip install pandas=1.3.0`.

## Running on Google Colab

The demos in this directory have been tested using Google Colab Pro. In order to run a demo, you can connect your Colab account to GitHub. To make use of provided datasets, you will also need to either connect Colab to your personal Google Drive, or directly upload data files to your Colab instance. 

### Step-by-Step Instructions

- Download the provided datasets from this [Google Drive Link](https://drive.google.com/drive/folders/1X-CgvkQKpatdPPrAYnWaeGmhA-daLJGr?usp=sharing).
- Upload these files to your personal Google Drive, noting their location.
- Download demo notebooks from this GitHub directory.
- Navigate to [Google Colab](https://colab.research.google.com/)
- From the **File** menu, select **Upload notebook**. You will be prompted to sign in to a Google account.
- Navigate to the **GitHub** tab. You will be prompted to connect Colab with your GitHub account. 
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

*In development*: N-BEATS, DeepAR, and Temporal Fusion Transformer applied to exchange rate forecasting.

### Demo 3: Rolling Cross-Validation

*In development*: Alternative data splitting and experimental design using rolling cross-validation and NeuralProphet. Applied to exchange rate forecasting.
