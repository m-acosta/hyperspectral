from PIL import Image
import pandas as pd
import os
import re
import json

def gps_to_pixels(latitude, longitude):
    x_pixel = int((longitude - min_longitude) * pixel_per_degree_lon)
    y_pixel = int((max_latitude - latitude) * pixel_per_degree_lat)
    return x_pixel, y_pixel

def pixels_to_gps(x_pixel, y_pixel):
    longitude = (x_pixel / pixel_per_degree_lon) + min_longitude
    latitude = max_latitude - (y_pixel / pixel_per_degree_lat)
    return round(latitude, 6), round(longitude, 6)

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

def csv_to_pixels(df):
    with open('gpsMapping.txt', 'a') as f:
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
            f.write(f"Latitude: {latitude}, Longitude: {longitude}, Pixels: {x_pixel}, {y_pixel}\n")

folder_path = './index_images'
files = os.listdir(folder_path)

csv_path = 'rounded scout data farm vineyard 2023.csv'
df = pd.read_csv(csv_path)
# df = df.round(6)
# df.to_csv('rounded scout data farm vineyard 2023.csv', index=False)

dictionary_mapping = {}

with open('boundingBoxMapping.json', 'r') as file:
    dictionary_mapping = json.load(file)

dictionary_mapping = {tuple(eval(k)): v for k, v in dictionary_mapping.items()}

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

    averages = []

    # Iterate through bounding boxes to find average value
    for key, value in dictionary_mapping.items():
        x_pixels, y_pixels = key
        latitude, longitude = pixels_to_gps(x_pixels, y_pixels)
        row_matches = df[(df['Latitude'] == latitude) & (df['Longitude'] == longitude)]

        x_min = value[1]
        y_min = value[2]
        width = value[3]
        height = value[4]

        pixel_values = []

        # Define the percentage of the inner region to consider (k percent)
        k_percent = 1

        # Calculate the size of the inner square based on the given percentage
        inner_width = int(width * k_percent)
        inner_height = int(height * k_percent)

        # Calculate the starting point for the inner square
        inner_x_start = int(x_min + (width - inner_width) / 2)
        inner_y_start = int(y_min + (height - inner_height) / 2)

        # Loop over the inner region of the square
        for x in range(inner_x_start, inner_x_start + inner_width):
            for y in range(inner_y_start, inner_y_start + inner_height):
                r, g, b = heatmap_image.getpixel((x, y))
                rgb_value = (r, g, b)
                scale = 2 * g / 255 - 1
                pixel_values.append(scale)

        # Take the average of the top k values
        average = sum(pixel_values) / len(pixel_values)
        for index, row in row_matches.iterrows():
            df.at[index, extracted_string] = average
        averages.append(average)

df.to_csv(csv_path, index=False)