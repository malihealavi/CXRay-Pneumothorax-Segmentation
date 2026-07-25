import pydicom
import numpy as np

def normalizer(dicom_path):
    """
    Reads a DICOM file, applies critical medical image normalization, 
    and converts it to a 3-channel RGB image.
    """
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array.astype(np.float32)
   
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = np.amax(image) - image
        
   
    image_min = np.min(image)
    image_max = np.max(image)
    if image_max - image_min > 0:
        image = (image - image_min) / (image_max - image_min) * 255.0
    else:
        image = np.zeros_like(image)
        
    image = image.astype(np.uint8)

    return image