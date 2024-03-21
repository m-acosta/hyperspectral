from PIL import Image
import pandas as pd
import os
import re
import json

def gps_to_pixels(latitude, longitude):
    x_pixel = int((longitude - min_longitude) * pixel_per_degree_lon)
    y_pixel = int((max_latitude - latitude) * pixel_per_degree_lat)
    return x_pixel, y_pixel

def dms_to_decimal(degrees, minutes, direction):
    decimal_degrees = float(degrees) + float(minutes) / 60
    if direction in ['S', 'W']:
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def parse_gps_coordinates(gps_string):
    direction = gps_string[0]  # Extract the direction (N, S, E, W)
    coordinates = gps_string[1:].split()  # Split the rest into degrees and minutes

    degrees = coordinates[0][:]  # Extract degrees, excluding the degree symbol
    minutes = coordinates[1]  # Extract minutes

    return dms_to_decimal(degrees, minutes, direction)

folder_path = './index_images'
files = os.listdir(folder_path)

csv_path = 'scout data farm vineyard august 2023.csv'
df = pd.read_csv(csv_path)

with open('gpsMonitor.json') as f:
    monitorData = json.load(f)

bounds = monitorData["polygons"][0]["array"]

min_longitude = min(bounds[::2])
max_longitude = max(bounds[::2])
min_latitude = min(bounds[1::2])
max_latitude = max(bounds[1::2])

lat_range = max_latitude - min_latitude
lon_range = max_longitude - min_longitude

for file_name in files:
    new_column_data = []
    image_path = folder_path + '/' + file_name
    
    # Extract the name of the index
    pattern = r'_or_(\w+)\.png'
    match = re.search(pattern, image_path)
    if match:
        extracted_string = match.group(1)
    else:
        print("No match found.")

    heatmap_image = Image.open(image_path).convert('RGB')
    image_width, image_height = heatmap_image.size

    pixel_per_degree_lat = image_height / lat_range
    pixel_per_degree_lon = image_width / lon_range
    
    # Convert the DataFrame to a CSV-formatted string
    csv_string = df.to_csv(index=False)

    # Split the CSV-formatted string into lines
    csv_lines = csv_string.splitlines()

    # Avoid the header lines
    header_line = csv_lines.pop(0)

    # Iterate through the lines
    for line in csv_lines:
        fields = line.split(',')
        latitude = float(fields[0])
        longitude = float(fields[1])

        # Translate GPS coordinates to pixels
        x_pixel, y_pixel = gps_to_pixels(latitude, longitude)

        if (x_pixel < image_width and y_pixel < image_height):
            r, g, b = heatmap_image.getpixel((x_pixel, y_pixel))
            rgb_value = (r, g, b)
            scale = 2 * g / 255 - 1
            new_column_data.append(scale)
        else:
            new_column_data.append(-2)
    
    df[extracted_string] = new_column_data

df.to_csv(csv_path, index=False)