import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Titel der App
st.title("👕 Kleidungs-KI")
st.write("Lade ein Bild hoch und die KI erkennt: Jeans, Shirt oder Hose")

# Modell laden
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("keras_model.h5")
    return model

model = load_model()

# Labels laden
def load_labels():
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels()

# Bild hochladen
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten (Teachable Machine nutzt meist 224x224)
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 127.5 - 1
    img_array = np.expand_dims(img_array, axis=0)

    # Vorhersage
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    st.subheader("🔎 Ergebnis:")
    st.write(f"**Klasse:** {labels[index]}")
    st.write(f"**Sicherheit:** {confidence * 100:.2f}%")
