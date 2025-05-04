import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random

def show_random_image_with_bbox(df, images_folder, bbox_column='bbox_scaled', 
                                file_column='file_name', label_column='name'):
    """
    Muestra aleatoriamente una imagen del dataframe con su bbox dibujado.
    """
    valid_df = df[df[bbox_column].notnull()]
    sample = valid_df.sample(1).iloc[0]
    image_path = os.path.join(images_folder, sample[file_column])

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    x0, y0, width, height = map(float, sample[bbox_column])
    x1, y1 = x0 + width, y0 + height

    draw.rectangle([x0, y0, x1, y1], outline='red', width=3)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Category: {sample[label_column]}")
    plt.axis('off')
    plt.show()

def show_image_with_multi_bbox(df, images_folder, file_name_column='file_name',
                               bbox_column='bbox_scaled', label_column='name', file_name=None):
    """
    Muestra una imagen (aleatoria si no se especifica) con todos sus bboxes dibujados.
    
    Parameters:
    - df: pandas.DataFrame con bboxes escalados.
    - images_folder: ruta a la carpeta de imágenes.
    - file_name_column: columna con el nombre del archivo.
    - bbox_column: columna con las cajas escaladas.
    - label_column: columna con las etiquetas.
    - file_name: (opcional) nombre de archivo específico. Si None, selecciona aleatorio.
    """
    valid_df = df[df[bbox_column].notnull()]
    
    if file_name is None:
        multi_bbox_files = valid_df.groupby(file_name_column).size()
        multi_bbox_files = multi_bbox_files[multi_bbox_files > 1].index
        if len(multi_bbox_files) == 0:
            print("No hay imágenes con múltiples bboxes.")
            return
        file_name = random.choice(multi_bbox_files)
    
    image_df = valid_df[valid_df[file_name_column] == file_name]
    
    if image_df.empty:
        print(f"No se encontraron bboxes para la imagen {file_name}")
        return
    
    image_path = os.path.join(images_folder, file_name)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    for _, row in image_df.iterrows():
        x0, y0, width, height = map(float, row[bbox_column])
        x1, y1 = x0 + width, y0 + height
        draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
        draw.text((x0, y0), row[label_column], fill='red')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"File: {file_name}")
    plt.axis('off')
    plt.show()

import os
import random
from PIL import Image as PILImage, ImageDraw
import matplotlib.pyplot as plt

def show_random_image_by_class(df, class_name, images_folder, bbox_column='bbox_scaled',
                               file_column='file_name', label_column='name'):
    """
    Muestra una imagen aleatoria de una clase específica del dataframe con su bbox dibujado.
    
    Parameters:
    - class_name: nombre exacto de la clase que quieres visualizar.
    - images_folder: carpeta donde están las imágenes.
    """
    filtered_df = df[(df[label_column] == class_name) & df[bbox_column].notnull()]
    if filtered_df.empty:
        print(f"No hay imágenes para la clase '{class_name}'.")
        return

    sample = filtered_df.sample(1).iloc[0]
    image_path = os.path.join(images_folder, sample[file_column])

    img = PILImage.open(image_path)
    draw = ImageDraw.Draw(img)

    x0, y0, width, height = map(float, sample[bbox_column])
    x1, y1 = x0 + width, y0 + height

    draw.rectangle([x0, y0, x1, y1], outline='red', width=3)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Category: {sample[label_column]}")
    plt.axis('off')
    plt.show()

def plot_training_history(history, title_prefix=""):
    """
    Grafica loss y accuracy de entrenamiento y validación.
    
    Args:
        history: history object (keras.callbacks.History) después de model.fit().
        title_prefix: (opcional) texto para añadir al inicio de los títulos de las gráficas.
    """
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'{title_prefix} Loss: Train vs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title(f'{title_prefix} Accuracy: Train vs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

