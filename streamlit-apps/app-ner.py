import streamlit as st

from simpletransformers.ner.ner_model import NERModel

from typing import Any

def load_model() -> NERModel:
    """Loads a pre-trained model"""
    return NERModel("bert", "../ner/outputs/", use_cuda=False, args={})

st.markdown("# Ejemplo de app con Streamlit que hace uso de modelos de NLP 🤖")

st.markdown("Usando la librería [`simpletransformers`](https://github.com/ThilinaRajapakse/simpletransformers) y colleciones de datos libres, hemos ajustado dos modelos de NLP diferentes:\n\n- uno para reconocimiento de entidades,\n- otro para clasificación de sentimiento.\n\n")

model = load_model()

st.markdown("## Reconocimiento de entidades")

text = None
text = st.text_input("Escribe una frase en inglés aquí:")
predictions, _raw_outputs = model.predict([text])

if text:
    st.markdown(f"Texto: *{text}*")
    st.write(predictions)

