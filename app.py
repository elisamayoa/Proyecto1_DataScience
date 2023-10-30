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


uploads  = st.file_uploader('choose an image to predict', type=['png', 'jpg', 'tif'])

X_test = None

if uploads is not None:
    image_bytes = uploads.read()
    uploaded_filename = uploads.name
    image_stream = BytesIO(image_bytes)

    img = Image.open(image_stream)
    
    img_path = f'images/{uploaded_filename}'
    img.save(img_path)
    
    
    _, imagen = preprocess(img_path)
    if imagen is not None:  # Añadir solo si la imagen fue preprocesada correctamente
        X_test = imagen
        predicciones = model.predict(X_test)
        print(predicciones)
