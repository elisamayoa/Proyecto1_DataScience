import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from functions import *
from PIL import Image
from io import BytesIO





st.header("Blood clot Prediction App")

model = keras.models.load_model('model.h5', custom_objects={'f1_score': f1_score})



options = ["None", "Image 1", "Image 2", "Image 3", "Image 4", "All images"]
image_paths = ['None', 'images/00c058_0.tif', 'images/01adc5_0.tif', 'images/008e5c_0.tif', 'images/006388_0.tif', 'All']
selected_option = st.selectbox("Select one of the images to test the model:", options)

if selected_option == "None":
    pass
elif selected_option == "All images":
    image_names = [i.split('/')[1].split('.')[0] for i in image_paths[1:-1]]
    data = pd.read_csv('data/train.csv')
    labels = data[data['image_id'].isin(image_names)]
    labels
else:
    idx = options.index(selected_option)
    
    image = image_paths[idx]
    image_name = image.split('/')[1].split('.')[0]
    data = pd.read_csv('data/train.csv')
    label = data[data['image_id']==image_name]
    label
    image2process = preprocess(image)
    X_test = []
    if image2process:
        X_test.append(image2process[1])
        X_test=np.array(X_test)[:, :, :, :3]
        predictions = model.predict(X_test)
        predicciones_clases = (predictions > 0.5).astype(int).reshape(-1)

        print('\naaa',predictions, predicciones_clases)
        
    st.button('hola')
    
    
    
    
    
    
    
# if uploads is not None:
#     image_bytes = uploads.read()
#     uploaded_filename = uploads.name
#     image_stream = BytesIO(image_bytes)

#     img = Image.open(image_stream)
    
#     img_path = f'images/{uploaded_filename}'
#     img.save(img_path)
    
    
#     _, imagen = preprocess(img_path)
#     if imagen is not None:  # Añadir solo si la imagen fue preprocesada correctamente
#         X_test = imagen
#         predicciones = model.predict(X_test)
#         print(predicciones)
