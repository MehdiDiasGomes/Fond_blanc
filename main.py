import cv2
import numpy as np
import os

def adjust_image(image_path, output_path):
    # Charger l'image
    image = cv2.imread(image_path)

    # Convertir l'image en espace de couleur LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Définir une zone de référence pour le "point blanc" en bas à gauche de l'image
    height, width = lab_image.shape[:2]
    white_ref_region = lab_image[height-10:height, 0:10]  # Zone de 10 pixels de hauteur en bas à gauche

    # Calculer la valeur moyenne de la luminosité dans cette région
    mean_luminance = np.mean(white_ref_region[:, :, 0])

    # Ajustement : ajouter un offset pour que le point blanc devienne effectivement blanc
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

# Traitement des images dans le dossier source
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        file_path = os.path.join(source_folder, filename)
        output_path = os.path.join(destination_folder, filename)
        try:
            adjust_image(file_path, output_path)
            print(f"Traitement de {filename} terminé.")
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")

print("Traitement terminé.")
