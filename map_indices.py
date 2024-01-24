from PIL import Image
import pandas as pd

def gps_to_pixels(latitude, longitude):
    x_pixel = int((longitude - min_longitude) * pixel_per_degree_lon)
    y_pixel = int((latitude - min_latitude) * pixel_per_degree_lat)
    return x_pixel, y_pixel

image_path = 'multi_or_NDVI.png'
heatmap_image = Image.open(image_path)
image_width, image_height = heatmap_image.size

# hard coded values based on the target image
min_latitude = 34.04868655101289
max_latitude = 34.05009333955231
min_longitude = -117.82098084065237
max_longitude = -117.82029084239056

lat_range = max_latitude - min_latitude
lon_range = max_longitude - min_longitude
pixel_per_degree_lat = image_height / lat_range
pixel_per_degree_lon = image_width / lon_range

csv_path = 'scout data farm vineyard august (reformatted).csv'
df = pd.read_csv(csv_path)

# Iterate through rows in the DataFrame and translate GPS to pixels
for index, row in df.iterrows():
    latitude = row['Latitude']
    longitude = row['Longitude']

    # Translate GPS coordinates to pixels
    x_pixel, y_pixel = gps_to_pixels(latitude, longitude)

    print(f"GPS: {latitude}, {longitude}  -->  Pixels: {x_pixel}, {y_pixel}")
