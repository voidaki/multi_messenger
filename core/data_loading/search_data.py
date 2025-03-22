from gwpy.table import Table
from pathlib import Path

gstlal_path_data = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/search_data_products/gstlal_allsky")
mbta_path_data = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/search_data_products/mbta_all_sky")
fars_gstlal = []
fars_mbta = []

#for file_path in gstlal_path_data.glob("*.xml"):
 #   table_data = Table.read(file_path, tablename="coinc_inspiral")
  #  far = table_data["combined_far"]
   # fars_gstlal.append(far)

for file_path in mbta_path_data.glob("*.xml"):
    table_data = Table.read(file_path, tablename="coinc_inspiral")
    far = table_data["false_alarm_rate"]
    fars_mbta.append(far)

print(f"Number of false alarm rates = {len(fars_mbta)}")
print(fars_mbta)


