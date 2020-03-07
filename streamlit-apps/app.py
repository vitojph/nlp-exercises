import streamlit as st

from simpletransformers.ner.ner_model import NERModel

def load_model(
    model_architecture: str, directory: str = "outputs/", use_cuda: bool = False, **kwargs
):
    """Loads a pre-trained model"""
    model = NERModel(model_architecture, directory, use_cuda=use_cuda, args=kwargs)
    return model

    
st.markdown("# Ejemplo de Reconocimiento de entidades")

model = load_model("bert", directory="ner-model")
if model:
    st.markdown("Modelo cargado")

text = None
text = st.text_input("Escribe tu texto aquí")
predictions, _raw_outputs = model.predict([text])

if text:
    st.markdown(f"Texto: *{text}*")
    st.write(predictions)
