
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

model_path = '/content/sample_data'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

thresholds = [0.52, 0.54, 0.61, 0.19, 0.18]
labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']

def classify_text(text):
    if not text.strip():
        return {label: "No input" for label in labels}

    inputs = tokenizer(
        text,
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

    results = {}
    for label, pred, prob in zip(labels, preds, probs):
        results[label] = f"{'Yes' if pred else 'No'} (Probability: {prob:.2f})"
    return results

iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=5, placeholder="Enter text to classify emotion..."),
    outputs=[gr.Label(label=label) for label in labels],
    title="Emotion Multi-label Classifier"
)

if __name__ == "__main__":
    iface.launch()
