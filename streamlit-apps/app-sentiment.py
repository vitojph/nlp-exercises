import streamlit as st

from simpletransformers.classification import ClassificationModel

from typing import Any

def load_model() -> ClassificationModel:
    """Loads a pre-trained model"""
    return ClassificationModel("distilbert", "../sentiment/outputs/", use_cuda=False, args={})

st.markdown("# Ejemplo de app con Streamlit que hace uso de modelos de NLP 🤖")

st.markdown("Usando la librería [`simpletransformers`](https://github.com/ThilinaRajapakse/simpletransformers) y colleciones de datos libres, hemos ajustado dos modelos de NLP diferentes:\n\n- uno para reconocimiento de entidades,\n- otro para clasificación de sentimiento.\n\n")

model = load_model()

st.markdown("## Análisis de sentimiento")


text = None
text = st.text_input("Escribe una frase en inglés aquí:")
predictions, _raw_outputs = model.predict([text])

if text:
    st.markdown(f"Texto: *{text}*")
    if predictions[0] == 1:
        st.write(f"Positivo")
    else:
        st.write(f"Negativo")

