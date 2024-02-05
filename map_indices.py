from PIL import Image
import pandas as pd
import os
import re

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

csv_path = 'scout data farm vineyard.csv'
df = pd.read_csv(csv_path)

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

    # Hard coded values based on the target image
    min_latitude = 34.04868655101289
    max_latitude = 34.05009333955231
    min_longitude = -117.82098084065237
    max_longitude = -117.82029084239056

    lat_range = max_latitude - min_latitude
    lon_range = max_longitude - min_longitude
    pixel_per_degree_lat = image_height / lat_range
    pixel_per_degree_lon = image_width / lon_range

    
    # Convert the DataFrame to a CSV-formatted string
    csv_string = df.to_csv(index=False)

    # Split the CSV-formatted string into lines
    csv_lines = csv_string.splitlines()

    # Iterate through the lines
    for line in csv_lines:
        # Avoid the header lines
        if line[0] != 'N':
            continue

        fields = line.split(',')
        raw_latitude = fields[0]
        raw_longitude = fields[1]

        latitude = parse_gps_coordinates(raw_latitude)
        longitude = parse_gps_coordinates(raw_longitude)

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