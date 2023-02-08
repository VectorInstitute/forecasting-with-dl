<a href="https://vectorinstitute.ai/"><img src="../media-assets-do-not-merge/vector-logo-black.svg?raw=true)" width="175" align="right" /></a>

# Forecasting Bootcamp

This repository contains demos and reference implementations for a variety of forecasting techniques that will be highlighted at Vector Institute's upcoming Forecasting Using Deep Learning Bootcamp in March 2022.

## Accessing Data

During the bootcamp, all reference datasets are available on the Vector cluster at ```/ssd003/projects/forecasting_bootcamp/bootcamp_datasets/```

For external use, we are providing the following link for downloading datasets:
- [Google Drive Link](https://drive.google.com/drive/folders/1X-CgvkQKpatdPPrAYnWaeGmhA-daLJGr?usp=sharing)

For a brief description about each dataset and their format you can refer to the table below. Note that very large datasets (e.g. WeatherBench) are not included, and that more datasets will be added over time.

| Dataset      | On Vector Cluster (Y/N) | On Google Drive (Y/N) | Description | Format |
| :------------ | :----------------- | :--------------- | :----------- | :---- | 
| [Canadian Weather Station Data](https://climate-change.canada.ca/climate-data/#/daily-climate-data)        | Y            | Y                 | Historical daily temperature and precipitation data for >100 weather stations in Canada since 1991.             | 4 features per weather station, 107 weather stations, 11697 daily observations, 107x4 possible targets |
| [Bank of Canada (BoC) Exchange Rate](https://www.bankofcanada.ca/rates/exchange/legacy-noon-and-closing-rates/) | Y | Y | Historical daily exchange rates between CAD and multiple currencies from 2007 to 2017. | 12 currencies (CAD to X exchange rate), 3651 daily observations |
| Electricity Consumption | Y | Y | Hourly electricity consumption data for 320 customers. | 320 customers, hourly observations of consumption per customer, 26304 observations, any column(s) could be used as targets |
| Road Traffic Occupancy | Y | Y | Hourly road occupancy data from 862 sensors from San Francisco / Bay Area freeways. | 861 traffic sensors, 17544 hourly observations, any column(s) as targets |
| Electricity Transformer Temperature (ETT) | Y | Y | Predicting the temperature of electricity transformers using hourly or every 15 minute data over two years. | Two versions: hourly and every 15 minutes observations (17420 and 69680 respectively), 6 features and 1 target. |
| Influenza-Like Illness Patient Ratios | Y | Y | Weekly case incidence rates of influenza-like illness reported to the Centers for Disease control, nationally, between 2002 and 2020. | 6 features (reporting statistics), 1 possible output (number of cases), 966 weekly observations |
| [Walmart M5 Retail Product Sales](https://www.kaggle.com/c/m5-forecasting-accuracy) | Y | Y | Individual product-level sales data for several products at Walmart, organized hierarchically. Curated list of resources: [link](https://docs.google.com/document/d/1ZRY5AK8Ox4SYFHPqlVJC0_YEwlHU2a-AQkaZKm9ZIes/edit?usp=sharing)| 42840 Hierarchical Time Series stemming from 3049 individual products at 10 Walmart accross 3 states over the span of 1941 days. |
| [WeatherBench](https://github.com/pangeo-data/WeatherBench) | Y | N | Global scale **spatiotemporal** weather forecasting dataset. | Low resolution world wide atmospheric data|
| ConnexOntario Call Volumes | Y | N | Metadata about >500K contacts to mental health services referral program from 2015 to 2020. | Data from Connex Ontario mental health line including date, location, substance abuse, mental health status etc. |
| [(Grocery) Store Sales (Corporación Favorita)](https://www.kaggle.com/c/store-sales-time-series-forecasting/data) | Y | Y | In this competition, you will predict sales for product families sold at Favorita stores located in Ecuador. The training data includes dates, store and product information, whether that item was being promoted, as well as the sales numbers. Additional files include supplementary information that may be useful in building your models. | 33 product families, 54 stores, 3 features (product sales data), 1 possible output (target sales)|
| Economic Data with Food CPI | Y | Y | Monthly observations of economic variables from FRED and StatCan, 1986 to 2021, used in forecasting models for Canada's Food Price Report. | 332 economic variables with 430 monthly values, any column(s) as targets. |

## Models
The demos contain reference applications of the following methods. 

| Model      | Lagged Covariates | Future Covariates | Multiple Targets | Probabilistic | Interpretable |
| :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | 
| [Prophet](https://facebook.github.io/prophet/)        |        | &check;     |            |  |&check; | 
| [Neural Prophet](https://neuralprophet.com/html/index.html) | &check; | &check; |  |  | &check; |
| [NBEATS](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.nbeats.NBeats.html) |  |  | &check; |  | &check; |
| [DeepAR](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.deepar.DeepAR.html) | | &check; | &check; | &check; |  |
| [Autoformer](https://arxiv.org/abs/2106.13008) | &check; | &check; | &check; |  | &check;|
| [NHITS](https://arxiv.org/abs/2201.12886) |   &check;|  &check; | &check; |  | &check;|


## Demos 
The demos for the bootcamp are available in the following directories:
- **intro_to_forecasting:** Two notebooks that overview the basics for time series analysis and time series forecasting. 
- **demos:** Outlines the application of Prophet, Neural Prophet, NBEATS, DeepAR and simple baseline methods to forecast exhange rates. The focus of these demos is to explore univariate forecasting. Lagged and/or future covariates are leveraged by methods that support them. Additionally, techniques for cross-validation are discussed. 
- **multivariate_demos:** Outlines the application of NBEATS and DeepAR to forecast hourly electricity consumption for a set of households based on past observations. The focus of this series of demos is to explore DeepAR and NBEATS in the multivariate forecasting setting. 
- **multivariate_lstf_demos:** Outlines the application of Autoformer and NHITS to forecasting a set of traffic sensors. The focus of these demos is too explore a multivariate time series forecasting problem where the desired input and ouput sequences are very large. 


## Main and dev branches

The AI Engineering team are using dev branches (e.g. [dev_demos](https://github.com/VectorInstitute/forecasting-bootcamp/tree/dev_demos)) to indicate works-in-progress. We invite you to view and begin experimenting with these resources, but please note that all material currently in development is subject to be modified, perhaps significantly, ahead of the bootcamp. 

## Environment 
In order to configure a Python 3.8 environment with the appropriate packages to run the demos, use the following sequence of commands on the command line:
- `conda create -n forecasting python=3.8`
- `conda activate forecasting`
- `conda install pip`
- `pip install -r forecasting_requirements.txt --user`

*NOTE*: If you are going to be using notebooks launched on our cluster, there is an issue with kernels associated with conda environments and cluster launched Jupyter notebooks. Thus, you should create an environment through `python3 -m venv` and source that in your slurm launch script or source from our prebuilt environment with `source /ssd003/projects/aieng/public/forecasting_unified/bin/activate`.
