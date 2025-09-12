import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
st.title('Bem-vindo ao reconhecimento de emoções de áudio por inteligência artificial! 😃')

# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Salvar temporariamente e reproduzir o áudio enviado
    st.audio(data=uploaded_file, format=uploaded_file.type, loop=False)

    # Extrair features
    # Code here

    # Normalizar os dados com o scaler treinado
    # Code here

    # Ajustar formato para o modelo
    # Code here

    # Fazer a predição
    # Code here

    # Exibir o resultado
    # Code here

    # Exibir probabilidades (gráfico de barras)
    # Code here

    # Remover o arquivo temporário
    # Code here
