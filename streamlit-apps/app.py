import streamlit as st

from simpletransformers.ner.ner_model import NERModel
from simpletransformers.classification import ClassificationModel

from typing import Any

def load_model(task: str) -> Any:
    """Loads a pre-trained model"""
    if task == "NER":
        model = NERModel("bert", "ner/outputs/")
    else:
        model = ClassificationModel("distilroberta", "sentiment/outputs/")
    return model

st.markdown("# Ejemplo de app de con modelos de NLP")

task = None
task = st.sidebar.selectbox(
        '¿Qué tarea quieres probar?',
        ('NER', 'Text Classification'))

if task:
    model = load_model(task)

if model:
    st.markdown("Modelo cargado")

text = None
text = st.text_input("Escribe tu texto aquí")
predictions, _raw_outputs = model.predict([text])

if text:
    st.markdown(f"Texto: *{text}*")
    st.write(predictions)
