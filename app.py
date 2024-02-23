import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model('model/')

def predict_dog_breed(image):
    # Prétraiter l'image
    image = image.resize((150, 150))
    image = image.convert('RGB')
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Faire la prédiction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    races = os.listdir("./Images")
    predicted_race = races[predicted_class]

    confidence = predictions[0][predicted_class]

    return predicted_race, confidence

def main():
    st.title("Identification de la race de chien")

    uploaded_image = st.file_uploader("Importer une image de chien", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Image importée', use_column_width=True)

        if st.button('Identifier la race'):
            predicted_race, confidence = predict_dog_breed(image)
            st.success(f"Race prédite: {predicted_race}, Confiance: {confidence:.2f}")

if __name__ == '__main__':
    main()
