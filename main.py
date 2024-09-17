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

# Dossiers source et de destination
source_folder = '/Users/mehdi/Desktop/Dev/Fond_blanc/Images'
destination_folder = '/Users/mehdi/Desktop/Dev/Fond_blanc/Images_Traites'

# Crée le dossier de destination s'il n'existe pas
os.makedirs(destination_folder, exist_ok=True)

# Coordonnées du point blanc que vous souhaitez définir
white_point_coords = (50, 50)  # Remplacez par les coordonnées souhaitées (x, y)

# Traitement des images dans le dossier source
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        file_path = os.path.join(source_folder, filename)
        output_path = os.path.join(destination_folder, filename)
        try:
            adjust_image(file_path, output_path, white_point_coords)
            print(f"Traitement de {filename} terminé.")
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")

print("Traitement terminé.")
