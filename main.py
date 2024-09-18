import cv2
import numpy as np
import os

def adjust_image(image_path, output_path, white_point_coords):
    image = cv2.imread(image_path)
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    white_ref_point = lab_image[white_point_coords[1], white_point_coords[0]]

    mean_luminance = white_ref_point[0]

    luminance_scale = 255 / mean_luminance
    lab_image[:, :, 0] = np.clip(lab_image[:, :, 0] * luminance_scale, 0, 255)

    corrected_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, corrected_image)

def is_image_suitable(image_path, white_area_threshold=0.8, min_contour_area=5000):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        return False

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_contour_area = sum(cv2.contourArea(cnt) for cnt in contours)

    total_area = image.shape[0] * image.shape[1]
    white_area_ratio = total_contour_area / total_area

    return white_area_ratio > white_area_threshold and total_contour_area > min_contour_area

source_folder = '/Users/mehdi/Desktop/Dev/Fond_blanc/Images'
destination_folder = '/Users/mehdi/Desktop/Dev/Fond_blanc/Images_Traites'
excluded_folder = '/Users/mehdi/Desktop/Dev/Fond_blanc/Images_Humains'

os.makedirs(destination_folder, exist_ok=True)
os.makedirs(excluded_folder, exist_ok=True)

white_point_coords = (50, 50)  

for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        file_path = os.path.join(source_folder, filename)
        output_path = os.path.join(destination_folder, filename)
        excluded_path = os.path.join(excluded_folder, filename)
        try:
            if is_image_suitable(file_path):
                adjust_image(file_path, output_path, white_point_coords)
                print(f"Traitement de {filename} terminé.")
            else:
                os.rename(file_path, excluded_path)
                print(f"{filename} exclue du traitement.")
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")

print("Traitement terminé.")
