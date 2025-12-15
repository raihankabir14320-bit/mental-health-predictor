import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import gdown
import zipfile

# --- CONFIGURATION ---
NUM_LABELS = 7 
CLASS_NAMES = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GOOGLE DRIVE FILE IDs (REPLACE THESE WITH YOUR ACTUAL IDs) ---
MENTAL_BERT_ID = '1Gy4kWP7TYFjCpDiEUVFDH-28f0oze1Un'
ROBERTA_ID = '11sbA4BZulFcn2EpIZ9QDs4jj-xYbB7fk'
# Ensure both files are shared as "Anyone with the link"

# --- MODEL DOWNLOAD AND LOAD FUNCTION ---
@st.cache_resource
def download_and_load_models():
    # Define a single temporary directory for all storage
    STORAGE_DIR = 'model_storage' 
    
    # 1. Setup Directories
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
        
    # Define the final paths where models will be loaded from
    bert_final_path = os.path.join(STORAGE_DIR, 'mental_bert_extracted')
    roberta_final_path = os.path.join(STORAGE_DIR, 'roberta_extracted')

    # 2. Download and Unzip Mental-BERT
    if not os.path.exists(bert_final_path):
        with st.spinner('Downloading Mental-BERT model...'):
            zip_output = os.path.join(STORAGE_DIR, 'mental_bert.zip')
            
            # Download the zip file
            gdown.download(id=MENTAL_BERT_ID, output=zip_output, quiet=True)
            
            # Extract the zip file contents into its dedicated folder
            with zipfile.ZipFile(zip_output, 'r') as zip_ref:
                # Extracts everything into bert_final_path
                zip_ref.extractall(bert_final_path)
            
            # Clean up the zip file to save space
            os.remove(zip_output)

    # 3. Download and Unzip RoBERTa
    if not os.path.exists(roberta_final_path):
        with st.spinner('Downloading RoBERTa model...'):
            zip_output = os.path.join(STORAGE_DIR, 'roberta.zip')
            
            # Download the zip file
            gdown.download(id=ROBERTA_ID, output=zip_output, quiet=True)
            
            # Extract the zip file contents into its dedicated folder
            with zipfile.ZipFile(zip_output, 'r') as zip_ref:
                zip_ref.extractall(roberta_final_path)
                
            # Clean up the zip file
            os.remove(zip_output)

    # 4. Load Models from the clean, extracted directories
    st.write("Loading models...")
    
    # Load Mental-BERT
    mental_bert_tokenizer = AutoTokenizer.from_pretrained(bert_final_path)
    mental_bert_model = AutoModelForSequenceClassification.from_pretrained(bert_final_path, num_labels=NUM_LABELS)
    mental_bert_model.to(DEVICE).eval()

    # Load RoBERTa
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_final_path)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_final_path)
    roberta_model.to(DEVICE).eval()
    
    return mental_bert_tokenizer, mental_bert_model, roberta_tokenizer, roberta_model

# Load models and tokenizers once using the cache
mental_bert_tokenizer, mental_bert_model, roberta_tokenizer, roberta_model = download_and_load_models()

# --- ENSEMBLE PREDICTION FUNCTION ---
# (Keep this function the same as before, as it uses the loaded models)
def ensemble_predict(text):
    """Tokenizes input and performs ensemble prediction by averaging probabilities."""
    
    # Mental-BERT Prediction
    inputs_bert = mental_bert_tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        logits_bert = mental_bert_model(**inputs_bert).logits
    probs_bert = F.softmax(logits_bert, dim=1)

    # RoBERTa Prediction
    inputs_roberta = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        logits_roberta = roberta_model(**inputs_roberta).logits
    probs_roberta = F.softmax(logits_roberta, dim=1)

    # Combine by Averaging Probabilities
    avg_probs = (probs_bert + probs_roberta) / 2
    
    # Get final prediction
    pred_id = avg_probs.argmax().item()
    confidence = avg_probs[0][pred_id].item()
    emotion = CLASS_NAMES[pred_id]
    
    return emotion, confidence, avg_probs

# --- Streamlit UI (Rest of your app code) ---
st.set_page_config(page_title="Ensemble Mental Health Classifier", layout="wide")

st.title("🧠 Ensemble Mental Health Classifier")
st.subheader("Combining Mental-BERT and RoBERTa for text classification.")

# Text input box for the user
user_input = st.text_area("Enter your text here:", 
                          "I feel overwhelmed and everything seems hopeless right now.",
                          height=150)

if st.button("Analyze Text"):
    if user_input:
        with st.spinner('Analyzing with ensemble model...'):
            emotion, confidence, all_probs = ensemble_predict(user_input)
            
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Predicted State", value=emotion, delta=f"{confidence:.2f} Confidence")
                
            with col2:
                st.write("---")
                import pandas as pd
                
                probs_df = all_probs.squeeze().cpu().numpy()
                probs_data = {
                    "State": CLASS_NAMES,
                    "Probability": [f"{p:.4f}" for p in probs_df]
                }
                
                df = pd.DataFrame(probs_data).sort_values(by="Probability", ascending=False)
                
                st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.warning("Please enter some text to analyze.")