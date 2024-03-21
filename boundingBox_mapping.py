import spectral
import matplotlib.pyplot as plt
from utils import *
import cv2
import matplotlib.pyplot as plt
import spectral


boxes = 'Yolo/multi_or_RGB.txt'
img = 'Yolo/multi_or_RGB.png'
mappings = 'gpsMapping.txt'
spectral_file = 'path to hdr file'
output_path = 'Mappings/'


# This function parses YOLO format bounding boxes and returns them in a list
def parse_yolo_format(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        boxes.append((class_id, x_center, y_center, width, height))

    # Convert YOLO coordinates to pixel coordinates
    yolo_boxes = [yolo_to_pixel(box, cv2.imread(img).shape) for box in boxes]

    return yolo_boxes

# This function will convert the YOLO box coordinates to the pixel coordinates
def yolo_to_pixel(yolo_box, image_shape):
    img_height, img_width = image_shape[:2]
    class_id, x_center, y_center, width, height = yolo_box

    x_center_pixel = int(x_center * img_width)
    y_center_pixel = int(y_center * img_height)
    width_pixel = int(width * img_width)
    height_pixel = int(height * img_height)

    # Calculate the top-left corner of the bounding box
    x_top_left = int(x_center_pixel - (width_pixel / 2))
    y_top_left = int(y_center_pixel - (height_pixel / 2))

    return class_id, x_top_left, y_top_left, width_pixel, height_pixel

# This function crops the bounding boxes from the image using pixel coordinates
def crop_bounding_boxes(image_path, boxes):
    # Read in the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Crop each bounding box from the image
    cropped_images = []
    for box in boxes:
        class_id, x, y, width, height = box
        crop_img = image[y:y+height, x:x+width]
        cropped_images.append((class_id, crop_img))

    return cropped_images


def show_images(cropped_images):
    n = len(cropped_images)
    cols = 3  # You can change the number of columns based on your preference
    rows = n // cols + (n % cols > 0)

    fig = plt.figure(figsize=(cols * 4, rows * 4))

    for i, (class_id, img) in enumerate(cropped_images, start=1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Class ID: {class_id}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def find_closest_bbox(target_pixel, bboxes):
    closest_bbox = None
    min_distance = float('inf')
    for bbox in bboxes:
        # Extract top-left coordinates, width, and height
        class_id, x_top_left, y_top_left, width, height = bbox
        
        # Calculate the center of the bbox
        x_center = x_top_left + width / 2
        y_center = y_top_left + height / 2
        
        # Calculate the Euclidean distance from the bbox center to the target pixel
        distance = ((x_center - target_pixel[0]) ** 2 + (y_center - target_pixel[1]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_bbox = bbox
    return closest_bbox


def extract_pixel_values(file_path, image_path):
    # List to store the pixel values
    pixel_values = []

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Open the file for reading
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line by the word 'Pixels: ' to isolate the pixel values part
            parts = line.split('Pixels: ')
            if len(parts) > 1:
                # Split the second part by comma to separate the pixel values and convert them to integers
                pixels_str = parts[1].split(', ')
                x, y = int(pixels_str[0]), int(pixels_str[1])
                
                # Check if the pixel coordinates are within the image dimensions
                if x < width and y < height:
                    # Ensure you check if the pixel is not fully black or white(it is in the interested area)
                    if not all(val == 0 or val == 255 for val in image[y, x]):
                        pixel_values.append([x, y])

    
    return pixel_values


def draw_bounding_boxes(image_path, boxes, points):
    # Read in the original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Convert YOLO coordinates to pixel coordinates
    # pixel_boxes = [yolo_to_pixel(box, image.shape) for box in boxes]

    # Draw each bounding box on the image
    for box in boxes:
        class_id, x_top_left, y_top_left, width, height = box
        top_left = (x_top_left, y_top_left)
        bottom_right = (x_top_left + width, y_top_left + height)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  

    for point in points:
        cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)  

    return image


def crop_hyperspectral_bounding_boxes(image_array, boxes, save_path):
    # Convert YOLO coordinates to pixel coordinates
    pixel_boxes = [yolo_to_pixel(box, image_array.shape) for box in boxes]

    # Crop each bounding box from the hyperspectral image
    for idx, box in enumerate(pixel_boxes):
        class_id, x, y, width, height = box
        crop_img = image_array[y:y+height, x:x+width, :]
        
        # Save the cropped hyperspectral image
        cropped_filename = f"{save_path}/cropped_{idx}_class_{class_id}.hdr"
        spectral.envi.save_image(cropped_filename, crop_img, dtype='float32', force=True)


def main():

    yolo_boxes = parse_yolo_format(boxes)

    # crops = crop_bounding_boxes(img, yolo_boxes)
    # show_images(crops)
    # hyperspectral_image_path = spectral_file
    # imgspec = spectral.open_image(hyperspectral_image_path)
    # data = np.array(imgspec.load())
    # print(data.shape)
    # crop_hyperspectral_bounding_boxes(data, yolo_boxes, output_path)


    points = extract_pixel_values(mappings, img)
    image_with_boxes = draw_bounding_boxes(img, yolo_boxes, points)
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    save_img(image_with_boxes_rgb, None, "Image with Bounding Boxes", output_path + "image_with_boxes_gps.jpg")

    cbx = []
    for i in points:
        cbx.append(find_closest_bbox(i, yolo_boxes))

    image_with_boxes = draw_bounding_boxes(img, cbx, [])
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    save_img(image_with_boxes_rgb, None, "Image with Bounding Boxes", output_path + "image_with_mapping.jpg")

if __name__ == '__main__':
    main()

