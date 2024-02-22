import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Charger le modèle pré-entraîné
# model = tf.keras.models.load_model('')

def predict_dog_breed(image):
    # Prétraiter l'image
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Faire la prédiction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence

def main():
    st.title("Identification de la race de chien")

    uploaded_image = st.file_uploader("Importer une image de chien", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Image importée', use_column_width=True)

        if st.button('Identifier la race'):
            predicted_class, confidence = predict_dog_breed(image)
            st.success(f"Race prédite: {predicted_class}, Confiance: {confidence:.2f}")

if __name__ == '__main__':
    main()
