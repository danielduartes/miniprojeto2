import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf
import tempfile
import pandas as pd

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler (iguais aos usados no notebook)
MODEL_PATH = "notebooks/models/modelo_sinistro.keras"
SCALER_PATH = "notebooks/models/scaler_sinistro.save"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções (iguais ao treino)
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features (igual ao notebook)
def extract_features(audio_path):
    data, sr = sf.read(audio_path)

    # Garantir mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    features = []

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data, pad=False), axis=1)
    features.extend(zcr)

    # Chroma STFT
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr), axis=1)
    features.extend(chroma)

    # MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr), axis=1)
    features.extend(mfccs)

    # RMS
    rms = np.mean(librosa.feature.rms(y=data, frame_length=2048, hop_length=512), axis=1)
    features.extend(rms)

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr), axis=1)
    features.extend(mel)

    # Garantir 162 features (igual no treino)
    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

st.title('Detector de emoções de áudio por IA! 😃')

uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"]
)

if uploaded_file is not None:
    # Salvar temporariamente o áudio
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.write(uploaded_file.getvalue())
    audio_path = temp_audio.name
    temp_audio.close()

    # Reproduzir o áudio
    st.audio(data=uploaded_file, format=uploaded_file.type, loop=False)

    # Extrair features
    features = extract_features(audio_path)

    # Normalizar os dados com o scaler treinado
    features = scaler.transform(features)

    # Expandir formato para o modelo
    features = np.expand_dims(features, axis=2)  # (1, 162, 1)

    # Fazer a predição
    pred = model.predict(features)
    pred_emotion = np.argmax(pred[0])
    emotion = EMOTIONS[pred_emotion]

    # Exibir o resultado
    st.write(f'**Emoção detectada:** {emotion}')

    # Exibir probabilidades (gráfico de barras)
    data_bar = pd.DataFrame({
        "Emoção": EMOTIONS, 
        "Probabilidade": pred[0] # pegando a primeira linha contendo as probabilidades de cada emoção
    })
    st.bar_chart(data_bar.set_index("Emoção"))

    # Remover o arquivo temporário
    os.remove(audio_path)