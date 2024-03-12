from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.interpolate import interp2d
import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

############################## DATA PREPROCESSING ##############################

def apply_pca(image, n_components=3):
    """
    Apply PCA to a hyperspectral image.

    Parameters:
    - image: np.array, the hyperspectral image with shape (height, width, bands).
    - n_components: int, the number of principal components to keep.

    Returns:
    - pca_image: np.array, the image transformed by PCA with shape (height, width, n_components).
    """
    # Flatten the image to (pixels, bands)
    original_shape = image.shape  # (height, width, bands)
    flattened_image = image.reshape(-1, original_shape[2])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(flattened_image)
    
    # Reshape the result back to (height, width, n_components)
    pca_image = pca_result.reshape(original_shape[0], original_shape[1], n_components)
    return pca_image
    
    

def image_interpolation(image, mask): 
    """Replaces outliers in image data using 2D interpolation.

    Args:
        image:  A NumPy array representing the image with outliers.
        mask:  A boolean NumPy array where True marks an outlier.

    Returns:
         NumPy array with outliers replaced via interpolation.
    """

    x, y = np.mgrid[:image.shape[0], :image.shape[1]] 
    valid_points = np.stack([x[~mask], y[~mask]], axis=-1)
    valid_values = image[~mask]

    interpolator = interp2d(valid_points[:, 0], valid_points[:, 1], valid_values)
    new_image = interpolator(x, y) 

    return new_image

def handle_outliers_image(image, method, outlier_mask):
    """Handles outliers in a NumPy array representing an image.

    Args:
        image:  A NumPy array representing the image.
        method: Handling method ('mean', 'median', 'remove', 'interpolate').
        outlier_mask: A boolean NumPy array where True marks an outlier.

    Returns:
        A NumPy array with outliers handled.
    """

    if method == 'mean':
        replacement_value = np.nanmean(image)
        image[outlier_mask] = replacement_value

    elif method == 'median':
        replacement_value = np.nanmedian(image)
        image[outlier_mask] = replacement_value

    elif method == 'remove':
        image[outlier_mask] = np.nan  

    elif method == 'interpolate':
        image = image_interpolation(image, outlier_mask) 

    else:
        raise ValueError('Invalid method')

    return image

def find_outliers(image, method):
    """Detects outliers in a NumPy array representing an image.

    Args:
        image:  A NumPy array representing the image.

    Returns:
        NumPy array where outlers are handled 
    """

    # Outlier Detection (Simple IQR-based)
    scaler = RobustScaler()
    image_reshaped = scaler.fit_transform(image.reshape(-1, 1))  
    outlier_mask_reshaped = abs(image_reshaped) > 2  # You can adjust this threshold
    outlier_mask = outlier_mask_reshaped.reshape(image.shape)
    image = handle_outliers_image(image, method, outlier_mask)
    return image



def min_max_scaler(data):
    """Performs min-max normalization on data using scikit-learn's MinMaxScaler

    Args:
        data: The NumPy array to normalize.

    Returns:
        The normalized NumPy array (values between 0 and 1).
    """
    scaler = MinMaxScaler() 
    return scaler.fit_transform(data)

def standard_scaler(data):
    """Standardizes data using scikit-learn's StandardScaler.

    Args:
        data: The NumPy array to standardize.

    Returns:
        The standardized NumPy array (zero mean, unit standard deviation).
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)

############################## save image ##############################


def show_image(vi, colormap, title=""):
    """
    Show image.
    
    :param vi: 2D numpy array of Vegetation Index data
    :param colormap: String of the colormap to use
    :param title: Title of the image
    """
    plt.imshow(vi, cmap=colormap)
    plt.colorbar()
    # plt.title(title)
    plt.axis('off')  # Optionally turn off the axis
    plt.show()



def save_img(vi, colormap, title, filename):
    # Check if the image has three channels
    if vi.ndim == 3 and vi.shape[2] == 3:
        # Image is already in three channels, save it directly
        normalized_band = (vi - np.min(vi)) / (np.max(vi) - np.min(vi)) * 255
        normalized_band = normalized_band.astype(np.uint8)  
        image = Image.fromarray(normalized_band)
    else:
        # Apply colormap to one-channel image
        normalized_band = (vi - np.min(vi)) / (np.max(vi) - np.min(vi))
        mapped_band = plt.get_cmap(colormap)(normalized_band)[:, :, :3]  # Exclude alpha channel if present
        image = Image.fromarray((mapped_band * 255).astype(np.uint8))

    # Save the image with high quality
    image.save(filename, 'JPEG', quality=100)


############################## FINDING WAVELENGHTH ##############################

def find_closest_wavelength_index(target, wavelengths):
    closest_wavelength = min(wavelengths, key=lambda x: abs(x - target))
    return closest_wavelength, wavelengths[closest_wavelength]