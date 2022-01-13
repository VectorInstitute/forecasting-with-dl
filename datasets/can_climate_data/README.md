# Canadian Weather Station Data

This dataset provides daily historical weather data from 109 stations across Canada between 1990 and 2022. For each station, the following features are included:

- Daily mean temperature (℃)
- Daily minimum temperature (℃)
- Daily maximum temperature (℃)
- Daily total precipitation - rainfall and water equivalent of snowfall (mm)

We selected 109 stations among 8447 total available from the Canadian [Climate Data Extraction Tool](https://climate-change.canada.ca/climate-data/#/daily-climate-data) that had greater than 95% coverage between 1990 and 2022 for the above features. 

The main dataset is stored in `weather_data.csv` and selected station metadata are stored in `station_metadata.csv`.

## Reconstructing the dataset

The data and metadata files can be reconstructed using the provided scripts.

Use `download_can_weather.py` to download the raw, per-station CSVs from the `weather.gc.ca` server. You need to specify a list of station IDs. Since we were unable to locate a list of valid station IDs, we iterated over numbers between 0 and 10,000 to locate and download valid files. Using this approach, we successfully downloaded 6,940 files. 

Once the raw data files are loaded, you can use `load_raw_data.ipynb` to load and process these files into a combined dataset. The notebook `data_explore.ipynb` can be used to inspect the data files.
