import gradio as gr
import joblib
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

RL_MODEL_PATH = 'LR/logistic_regression_model.pkl'
TFIDF_PATH = 'LR/tfidf_vectorizer.pkl'
BERT_FINETUNED_OPTUNA = './bert_finetuned_optuna/'

global rl_model
global tfidf_vectorizer

try:
    rl_model = joblib.load(RL_MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    rl_available = True
except FileNotFoundError:
    rl_available = False

global bert_tokenizer
global bert_model

bert_available = False

if os.path.isdir(BERT_FINETUNED_OPTUNA):
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_FINETUNED_OPTUNA)
        bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_FINETUNED_OPTUNA)
        bert_model.eval()
        bert_available = True
    except:
        pass

def format_output(prob_class_1, label):
    clase_predicha = "Depresivo (1)" if label == 1 else "No Depresivo (0)"
    prob_display = prob_class_1 if label == 1 else 1.0 - prob_class_1
    return f"Clase Predicha: **{clase_predicha}**\nProbabilidad de Depresión (Clase 1): {prob_class_1:.4f}"

def normalize_text_advanced(text):
    if pd.isna(text) or text == '':
        return ''
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        text = ' '.join(tokens)
        return text
    except:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def predict_rl(text):
    if not rl_available:
        return "El modelo de Regresión Logística no está disponible."
    text = normalize_text_advanced(text)
    try:
        text_vectorized = tfidf_vectorizer.transform([text])
        prob_class_1 = rl_model.predict_proba(text_vectorized)[0][1]
        prediction = rl_model.predict(text_vectorized)[0]
    except Exception as e:
        return f"Error durante la predicción RL: {e}"
    return format_output(prob_class_1, prediction)

def predict_bert(text):
    if not bert_available:
        return "El modelo DistilBERT no está disponible."
    try:
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
        prediction = np.argmax(probabilities).item()
        prob_class_1 = probabilities[1]
    except Exception as e:
        return f"Error durante la predicción BERT: {e}"
    return format_output(prob_class_1, prediction)

def predict_model(model_name, text_input):
    if not text_input or len(text_input.strip()) == 0:
        return "Por favor, ingresa algún texto para analizar."
    if model_name == "Regresión Logística + TF-IDF":
        return predict_rl(text_input)
    elif model_name == "DistilBERT (Recomendado)":
        return predict_bert(text_input)
    else:
        return "Modelo no válido."

with gr.Blocks(title="Clasificación de Depresión en Reddit") as demo:
    gr.Markdown(
        """
        # 🧠 Detección de Lenguaje Asociado a Depresión en Publicaciones de Reddit
        Selecciona un modelo, ingresa un texto y obtén la clasificación binaria (Depresivo/No Depresivo).
        """
    )
    
    model_selector = gr.Radio(
        choices=[
            "DistilBERT (Recomendado)",
            "Regresión Logística + TF-IDF"
        ],
        value="DistilBERT (Recomendado)",
        label="Selecciona el Modelo de Clasificación"
    )
    
    text_input = gr.Textbox(
        label="Ingresa la Publicación de Texto",
        placeholder="Escribe aquí tu publicación...",
        lines=5
    )
    
    submit_button = gr.Button("Analizar Texto")
    
    output_markdown = gr.Markdown(
        label="Resultado de la Clasificación",
        value="Esperando la entrada del usuario..."
    )
    
    submit_button.click(
        fn=predict_model,
        inputs=[model_selector, text_input],
        outputs=output_markdown
    )

if __name__ == "__main__":
    demo.launch()
