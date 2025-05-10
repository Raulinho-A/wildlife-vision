import numpy as np
import pandas as pd
from PIL import Image
import os
import albumentations as A
import cv2
from tqdm import tqdm

def rescale_bounding_boxes(df, target_width, target_height, bbox_column='bbox',
                           width_column='width', height_column='height',
                           output_column='bbox_scaled'):
    """
    Reescala bounding boxes a nuevas dimensiones objetivo.
    """
    original_width = df[width_column].iloc[0]
    original_height = df[height_column].iloc[0]
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    def scale_bbox(bbox):
        if bbox is None or isinstance(bbox, float) and np.isnan(bbox):
            return None 
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None 
        x0, y0, width, height = map(float, bbox)
        x1, y1 = x0 + width, y0 + height
        x0_scaled = x0 * scale_x
        y0_scaled = y0 * scale_y
        width_scaled = (x1 - x0) * scale_x
        height_scaled = (y1 - y0) * scale_y
        return [x0_scaled, y0_scaled, width_scaled, height_scaled]

    df[output_column] = df[bbox_column].apply(scale_bbox)
    return df

def save_recortes_by_class(df, images_folder, output_folder, bbox_column='bbox_scaled',
                           file_column='file_name', label_column='name'):
    """
    Recorta las imágenes según bbox y guarda en carpetas por clase.
    """
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving crops"):
        class_name = row[label_column]
        bbox = row[bbox_column]
        if bbox is None:
            continue
        file_name = row[file_column]
        
        class_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        img_path = os.path.join(images_folder, file_name)
        img = Image.open(img_path)
        
        x0, y0, width, height = map(float, bbox)
        x1, y1 = x0 + width, y0 + height
        cropped_img = img.crop((x0, y0, x1, y1))
        
        save_name = f"{os.path.splitext(file_name)[0]}_{row['id_ann']}.jpg"
        save_path = os.path.join(class_folder, save_name)
        if not os.path.exists(save_path):
            cropped_img.save(save_path)

def apply_static_augmentations_for_class(class_name, input_root, output_root, num_augmentations=2):
    """
    Aplica augmentations a una clase específica.
    
    Parameters:
    - class_name: nombre de la clase (carpeta dentro de input_root).
    - input_root: carpeta raíz de los recortes originales.
    - output_root: carpeta raíz para guardar augmentations.
    - num_augmentations: cuántas imágenes augmentadas generar por original.
    """
    input_folder = os.path.join(input_root, class_name)
    output_folder = os.path.join(output_root, class_name)
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.GaussNoise(p=0.2)
    ])
    
    os.makedirs(output_folder, exist_ok=True)
    
    for img_name in tqdm(os.listdir(input_folder), desc=f"Augmenting {class_name}"):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"No se pudo leer la imagen: {img_path}")
            continue
        
        for i in range(num_augmentations):
            augmented = transform(image=img)['image']
            save_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, save_name), augmented)

def count_files_per_class(base_folder):
    class_counts = {}
    for class_name in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_name)
        if os.path.isdir(class_path):
            num_files = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_name] = num_files

    sorted_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_counts

def apply_augmentations_for_class(class_name, input_root, output_root, transform, num_augmentations=2):
    """
    Aplica augmentations a una clase específica.

    Parameters:
    - class_name: nombre de la clase (carpeta dentro de input_root).
    - input_root: carpeta raíz de los recortes originales.
    - output_root: carpeta raíz para guardar augmentations.
    - transform: objeto albumentations.Compose con las transformaciones.
    - num_augmentations: cuántas imágenes augmentadas generar por original.
    """
    input_folder = os.path.join(input_root, class_name)
    output_folder = os.path.join(output_root, class_name)

    os.makedirs(output_folder, exist_ok=True)

    for img_name in tqdm(os.listdir(input_folder), desc=f"Augmenting {class_name}"):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"No se pudo leer la imagen: {img_path}")
            continue

        for i in range(num_augmentations):
            augmented = transform(image=img)['image']
            save_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, save_name), augmented)

def show_random_augmented_image_by_class(class_name, augmented_folder='bboxes\\augmented'):
    """
    Muestra aleatoriamente una imagen augmentada de la clase indicada.
    
    Parameters:
    - class_name: nombre de la carpeta/clase dentro de augmented_folder.
    - augmented_folder: carpeta raíz donde están las carpetas de augmentations.
    """
    class_folder = os.path.join(augmented_folder, class_name)
    if not os.path.exists(class_folder):
        print(f"Carpeta no encontrada: {class_folder}")
        return
    
    images = os.listdir(class_folder)
    if not images:
        print(f"No hay imágenes en {class_folder}")
        return
    
    selected_image = random.choice(images)
    img_path = os.path.join(class_folder, selected_image)
    
    img = Image.open(img_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Class: {class_name}\nFile: {selected_image}")
    plt.axis('off')
    plt.show()