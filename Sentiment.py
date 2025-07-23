import shap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import pipeline
import streamlit as st

st.set_page_config(layout="wide")

# Load sentiment analysis model


@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")


@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model


analyzer = load_pipeline()
tokenizer, model = load_model_and_tokenizer()
# Title and input
st.title("Sentiment Analysis App with SHAP Explanation")
user_input = st.text_input("Enter a sentence to analyze:")

# INPUT
if user_input:
    # Run prediction
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(probs).item()
    confidence = torch.max(probs).item()
    label_names = ['Negative', 'Neutral', 'Positive']

    st.write(f"**Label:** {label_names[prediction]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Token level logit contributions
    st.subheader("Token Contributions (logit scale)")
    token_logits = logits.detach().numpy()[0]
    data = {label_names[i]: token_logits[i] for i in range(len(label_names))}
    # fig, ax = plt.subplots()
    # sns.barplot(x=list(data.keys()), y=list(data.values()), ax=ax)
    # ax.set_title("Sentence-Level Logits")
    # st.pyplot(fig)

    # SHAP Explanation
    st.subheader("Explain Sentence Prediction with SHAP")


def f(x):
    # Convert NumPy array to list of strings if necessary
    if isinstance(x, np.ndarray):
        x = x.tolist()
    inputs = tokenizer(x, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
        probs = torch.nn.functional.softmax(output.logits, dim=1).numpy()
    return probs


explainer = shap.Explainer(f, shap.maskers.Text(tokenizer))
shap_values = explainer([user_input])
shap_values.output_names = ["Negative", "Neutral", "Positive"]

st.write("**SHAP Visualization:**")
shap_html = shap.plots.text(shap_values[0], display=False)
st.components.v1.html(shap.getjs(), height=0)
st.components.v1.html(shap_html, height=300)


# CSV Upload
st.subheader("Upload a CSV file with text data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.write(df.head())

    text_column = st.selectbox(
        "Select the column that contains text", df.columns)

    if st.button("Run Sentiment Analysis"):
        with st.spinner("Analyzing sentiments..."):
            results = analyzer(df[text_column].astype(str).tolist())

        df["Sentiment"] = [res["label"] for res in results]
        df["Confidence"] = [res["score"] for res in results]

        st.success("Analysis complete!")
        st.write(df.head())

        st.download_button(
            label="Download Results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_results.csv",
            mime="text/csv",
        )
