
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

model_path = '/content/model' 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

thresholds = [0.52, 0.54, 0.61, 0.19, 0.18]
labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']

st.title("Emotion Multi-label Classifier")

user_input = st.text_area("Enter text to classify:")

if st.button("Predict"):
    if user_input.strip():
        inputs = tokenizer(
            user_input,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        preds = (probs >= thresholds).astype(int)

        st.write("Predicted emotions:")
        for label, pred, prob in zip(labels, preds, probs):
            st.write(f"{label}: {'Yes' if pred else 'No'} (Probability: {prob:.2f})")
    else:
        st.write("Please enter some text to classify.")
