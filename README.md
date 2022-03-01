<a href="https://vectorinstitute.ai/"><img src="../media-assets-do-not-merge/vector-logo-black.svg?raw=true)" width="175" align="right" /></a>

# Forecasting Bootcamp

This repository contains demos and reference implementations for a variety of forecasting techniques that will be highlighted at Vector Institute's upcoming Forecasting Using Deep Learning Bootcamp in March 2022.

## Accessing Data

During the bootcamp, all reference datasets are available on the Vector cluster at ```/ssd003/projects/forecasting_bootcamp/bootcamp_datasets/```

For external use, we are providing the following link for downloading datasets:
- [Google Drive Link](https://drive.google.com/drive/folders/1X-CgvkQKpatdPPrAYnWaeGmhA-daLJGr?usp=sharing)

For a brief description about each dataset and their format you can refer to the table below. Note that very large datasets (e.g. WeatherBench) are not included, and that more datasets will be added over time.\

| Dataset      | On Vector Cluster (Y/N) | On Google Drive (Y/N) | Description | Format |
| :------------ | :----------------- | :--------------- | :----------- | :---- | 
| [Canadian Weather Station Data](https://climate-change.canada.ca/climate-data/#/daily-climate-data)        | Y            | Y                 | Historical daily temperature and precipitation data for >100 weather stations in Canada since 1991.             | 4 features per weather station, 107 weather stations, 11697 daily observations, 107x4 possible targets |
| [Bank of Canada (BoC) Exchange Rate](https://www.bankofcanada.ca/rates/exchange/legacy-noon-and-closing-rates/) | Y | Y | Historical daily exchange rates between CAD and multiple currencies from 2007 to 2017. | 12 currencies (CAD to X exchange rate), 3651 daily observations |
| Electricity Consumption | Y | N | Hourly electricity consumption data for 320 customers. | 320 customers, hourly observations of consumption per customer, 26304 observations, any column(s) could be used as targets |
| Road Traffic Occupancy | Y | N | Hourly road occupancy data from 862 sensors from San Francisco / Bay Area freeways. | 861 traffic sensors, 17544 hourly observations, any column(s) as targets |
| Electricity Transformer Temperature (ETT) | Y | N | Predicting the temperature of electricity transformers using hourly or every 15 minute data over two years. | Two versions: hourly and every 15 minutes observations (17420 and 69680 respectively), 6 features and 1 target. |
| Influenza-Like Illness Patient Ratios | Y | N | Weekly case incidence rates of influenza-like illness reported to the Centers for Disease control, nationally, between 2002 and 2020. | 6 features (reporting statistics), 1 possible output (number of cases), 966 weekly observations |
| [Walmart M5 Retail Product Sales](https://www.kaggle.com/c/m5-forecasting-accuracy) | Y | N | Individual product-level sales data for several products at Walmart, organized hierarchically. | Different datasets provided from walmart on sales such as date, frequency and sell prices |
| [WeatherBench](https://github.com/pangeo-data/WeatherBench) | Y | N | Global scale **spatiotemporal** weather forecasting dataset. | Low resolution world wide atmospheric data|
| ConnexOntario Call Volumes | Y | N | Metadata about >500K contacts to mental health services referral program from 2015 to 2020. | Data from Connex Ontario mental health line including date, location, substance abuse, mental health status ,and etc.|
| [(Grocery) Store Sales (Corporación Favorita)](https://www.kaggle.com/c/store-sales-time-series-forecasting/data) | Y | Y | In this competition, you will predict sales for the thousands of product families sold at Favorita stores located in Ecuador. The training data includes dates, store and product information, whether that item was being promoted, as well as the sales numbers. Additional files include supplementary information that may be useful in building your models. | 3 features (product sales data), 1 possible output (target sales)|


## Main and dev branches

The AI Engineering team are using dev branches (e.g. [dev_demos](https://github.com/VectorInstitute/forecasting-bootcamp/tree/dev_demos)) to indicate works-in-progress. We invite you to view and begin experimenting with these resources, but please note that all material currently in development is subject to be modified, perhaps significantly, ahead of the bootcamp. 
