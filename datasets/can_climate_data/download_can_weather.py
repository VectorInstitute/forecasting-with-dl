import os

if not os.path.exists("./climate_data"):
    os.mkdir("./climate_data")

station_ids = []

for station_id in station_ids:
	url = f"https://api.weather.gc.ca/collections/climate-daily/items?datetime=1990-01-01%2000:00:00/2022-01-10%2000:00:00&STN_ID={station_id}&sortby=PROVINCE_CODE,STN_ID,LOCAL_DATE&f=csv&limit=150000&startindex=0"
	try:
		command = f"""wget -O ./climate_data/station_{station_id}_data.csv "{url}" """
		os.system(command)	
	except Exception as e:
		print(f"Error processing station_id {station_id}.")
