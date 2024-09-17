import cv2
import numpy as np
import os

def adjust_image(image_path, output_path, white_point_coords):
    # Charger l'image
    image = cv2.imread(image_path)
    
    # Convertir l'image en espace de couleur LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Sélectionner le point blanc de référence à partir des coordonnées données
    white_ref_point = lab_image[white_point_coords[1], white_point_coords[0]]

    # Calculer la valeur de luminosité (L channel) du point blanc sélectionné
    mean_luminance = white_ref_point[0]

    # Ajustement : mettre à l'échelle la luminosité pour que ce point devienne blanc (L=255)
    luminance_scale = 255 / mean_luminance
    lab_image[:, :, 0] = np.clip(lab_image[:, :, 0] * luminance_scale, 0, 255)

    # Reconvertir en espace de couleur BGR
    corrected_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Sauvegarder l'image corrigée
    cv2.imwrite(output_path, corrected_image)

def is_image_suitable(image_path, white_area_threshold=0.8, min_contour_area=5000):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        return False

    # Convertir en espace de couleur HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir une plage de seuil pour détecter les pixels blancs
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # Créer un masque pour les pixels blancs
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    
    # Trouver les contours des zones blanches
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculer la surface totale des contours blancs
    total_contour_area = sum(cv2.contourArea(cnt) for cnt in contours)

    # Calculer la proportion de pixels blancs
    total_area = image.shape[0] * image.shape[1]
    white_area_ratio = total_contour_area / total_area

    # Vérifier si la proportion de pixels blancs dépasse le seuil
    # et si les contours blancs sont suffisamment grands
    return white_area_ratio > white_area_threshold and total_contour_area > min_contour_area

# Dossiers source, de destination et d'exclusion
source_folder = '/Users/mehdi/Desktop/Dev/Fond_blanc/Images'
destination_folder = '/Users/mehdi/Desktop/Dev/Fond_blanc/Images_Traites'
excluded_folder = '/Users/mehdi/Desktop/Dev/Fond_blanc/Images_Humains'

# Crée les dossiers de destination et d'exclusion s'ils n'existent pas
os.makedirs(destination_folder, exist_ok=True)
os.makedirs(excluded_folder, exist_ok=True)

# Coordonnées du point blanc que vous souhaitez définir
white_point_coords = (50, 50)  # Remplacez par les coordonnées souhaitées (x, y)

# Traitement des images dans le dossier source
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
                # Copier l'image non traitée dans le dossier d'exclusion
                os.rename(file_path, excluded_path)
                print(f"{filename} exclue du traitement.")
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")

print("Traitement terminé.")
