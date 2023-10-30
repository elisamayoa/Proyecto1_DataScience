
import keras.backend as K 
import numpy as np
from PIL import Image
import os




# Get the current working directory
current_directory = os.getcwd()

# Construct the relative path to the OpenSlide DLL directory
relative_path = os.path.join(current_directory, 'openslide-win64-20231011', 'bin')

# Set the OPENSLIDE_PATH to the relative path
OPENSLIDE_PATH = relative_path



if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import OpenSlide

def reduce_resolution(img, base_width=256):
    # Convertir la imagen a PIL si es necesario
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Calcular la proporción
    w_percent = base_width / float(img.size[0])
    h_size = int(float(img.size[1]) * float(w_percent))
    
    # Redimensionar
    img_resized = img.resize((base_width, h_size), Image.LANCZOS)
    
    return np.array(img_resized)



def preprocess(image_path):
    try:
        image = Image.open("images/linkedinpp.tif")

        print('aaaaaaaaaaaaaa')
        # Cargar la imagen con OpenSlide
        slide = OpenSlide(image_path)
        
        # Dimensiones de la imagen
        width, height = slide.dimensions
        print(width, height)
        # Dimensiones de la ventana
        window_width, window_height = 7500, 7500
        
        # Calcular las coordenadas de inicio para centrar la ventana
        start_x = (width - window_width) // 2
        start_y = (height - window_height) // 2
        
        # Leer la región centrada y redimensionarla
        image = slide.read_region((start_x, start_y), 0, (window_width, window_height))
        image = reduce_resolution(image)
        image = np.array(image)
        
        # Convertir la imagen a escala de grises
        imagen_gris = np.mean(image, axis=2)
        
        # Calcular el área blanca en la imagen
        area_blanca = np.sum(imagen_gris > 200)  # Consideramos como blanco los píxeles con intensidad mayor a 200
        
        # Calcular el área total de la imagen
        area_total = 256 * 256
        
        # Calcular la proporción de área blanca
        proporcion_area_blanca = area_blanca / area_total
        
        return proporcion_area_blanca, image  # También retornamos la imagen
    except Exception as e:
        print(f"Error al procesar la imagen {image_path}: {str(e)}")
        return None, None

def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val