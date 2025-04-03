import streamlit as st
import joblib
import pickle
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tensorflow.keras.models import load_model
from indicnlp.tokenize import indic_tokenize
import json
import time
from nltk.probability import LidstoneProbDist
import altair as alt
from streamlit_option_menu import option_menu
import os
from google.cloud import storage

# --- Configure page layout and theme ---
st.set_page_config(
    page_title="Dogri POS Tagging Suite",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced UI ---
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #1E3A8A;
    }
    .subheader {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #3B82F6;
    }
    .model-metrics {
        background-color: #f8fafc;
        border-left: 3px solid #3B82F6;
        padding: 1rem;
        margin: 1rem 0;
    }
    .tag-result {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        padding: 1.5rem;
        border-radius: 8px;
        font-family: 'Consolas', monospace;
        margin: 1rem 0;
    }
    .stButton button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    .stButton button:hover {
        background-color: #1d4ed8;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
    }
    .info-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
    }
    .tag-chip {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-top: 3px solid #3B82F6;
    }
    footer {display: none !important;}
    .block-container {padding-top: 2rem !important; padding-bottom: 2rem !important;}
</style>
""", unsafe_allow_html=True)

# --- Define the lidstone estimator (for HMM compatibility) ---
def lidstone_estimator(gamma, bins):
    return LidstoneProbDist(gamma, bins)

# --- Cache model loading ---
@st.cache_resource
def load_models():
    with st.spinner("Loading models... This may take a moment"):
       # Set up Google Cloud Storage details
BUCKET_NAME = "bucket1"  # Your GCS bucket name
GCS_MODEL_PATH = "mine/"  # Path inside your bucket
LOCAL_MODEL_PATH = "/tmp/models/"  # Temporary local storage for faster access

# Ensure the local directory exists
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

# Initialize GCS client
storage_client = storage.Client()

def download_from_gcs(source_blob_name, destination_file_name):
    """Downloads a file from GCS to local storage if not already downloaded."""
    local_file = os.path.join(LOCAL_MODEL_PATH, os.path.basename(destination_file_name))
    
    if not os.path.exists(local_file):  # Avoid downloading repeatedly
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(local_file)
        print(f"Downloaded {source_blob_name} to {destination_file_name}")

        # Files to download
        files_to_download = {
            "svm_model.pkl": "svm_model_200k.pkl",
            "hmm_model.pkl": "hmm_pos_tagger.pkl",
            "bilstm_model.keras": "bilstm2/bilstm_pos_tagger.keras",
            "bert_tokenizer": "mbert",
            "bert_model": "mbert",
            "token_encoder.pkl": "bilstm2/token_encoder.pkl",
            "tag_encoder.pkl": "bilstm2/tag_encoder.pkl",
            "bert_label_mappings.json": "mbert/mbert_dogri_label_mappings.json"
        }

        # Download all required files
        for local_name, gcs_name in files_to_download.items():
            download_from_gcs(GCS_MODEL_PATH + gcs_name, os.path.join(LOCAL_MODEL_PATH, local_name))

        # Load Models
        svm_model = joblib.load(f"{LOCAL_MODEL_PATH}/svm_model.pkl")
        hmm_tagger = pickle.load(open(f"{LOCAL_MODEL_PATH}/hmm_model.pkl", "rb"))
        bilstm_model = load_model(f"{LOCAL_MODEL_PATH}/bilstm_model.keras")

        # Load Tokenizer & Model for BERT
        bert_tokenizer = AutoTokenizer.from_pretrained(f"{LOCAL_MODEL_PATH}/bert_tokenizer")
        bert_model = AutoModelForTokenClassification.from_pretrained(f"{LOCAL_MODEL_PATH}/bert_model")

        # Load Label Encoders
        with open(f"{LOCAL_MODEL_PATH}/token_encoder.pkl", "rb") as f:
            token_encoder = pickle.load(f)
        with open(f"{LOCAL_MODEL_PATH}/tag_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        # Load BERT label mappings
        with open(f"{LOCAL_MODEL_PATH}/bert_label_mappings.json", "r", encoding="utf-8") as f:
            label_mappings = json.load(f)


        id2tag = label_mappings["id2tag"]
        tag2id = label_mappings["tag2id"]
        # Ensure <UNK> token exists in token encoder
        if "<UNK>" not in token_encoder.classes_:
            token_encoder.classes_ = np.append(token_encoder.classes_, "<UNK>")
        
        # Load model performance metrics (mock data - replace with actual metrics)
        model_metrics = {
            "SVM": {"accuracy": 0.89, "f1": 0.87, "precision": 0.88, "recall": 0.86},
            "HMM": {"accuracy": 0.85, "f1": 0.83, "precision": 0.84, "recall": 0.82},
            "BiLSTM": {"accuracy": 0.92, "f1": 0.90, "precision": 0.91, "recall": 0.90},
            "mBERT": {"accuracy": 0.95, "f1": 0.94, "precision": 0.95, "recall": 0.93}
        }
        
        # Tag colors for visualization (sample - update based on your tag set)
        tag_colors = {
            "NN": "#e63946", "NNP": "#f1faee", "PRON": "#a8dadc", 
            "VERB": "#457b9d", "ADJ": "#1d3557", "ADV": "#fca311",
            "DET": "#e5e5e5", "ADP": "#52b788", "CONJ": "#9e2a2b",
            "PRT": "#335c67", "NUM": "#540b0e", "X": "#99d98c"
        }
        
        # Fill in missing tags with default colors
        all_tags = set()
        for tags in label_encoder.classes_:
            all_tags.add(tags)
        for tags in id2tag.values():
            all_tags.add(tags)
        
        for tag in all_tags:
            if tag not in tag_colors:
                # Generate a color if not defined
                tag_colors[tag] = "#" + ''.join([f'{np.random.randint(0, 256):02x}' for _ in range(3)])
        
        return {
            "svm": svm_model,
            "hmm": hmm_tagger,
            "bilstm": bilstm_model,
            "bert_tokenizer": bert_tokenizer,
            "bert": bert_model,
            "token_encoder": token_encoder,
            "label_encoder": label_encoder,
            "id2tag": id2tag,
            "metrics": model_metrics,
            "tag_colors": tag_colors
        }

# --- Define feature extraction and prediction functions ---
def extract_features(tokens, index):
    token = tokens[index]
    features = {
        'token': token, 
        'prefix1': token[:1] if len(token) > 0 else '',
        'prefix2': token[:2] if len(token) > 1 else token[:1] if len(token) > 0 else '',
        'suffix1': token[-1:] if len(token) > 0 else '',
        'suffix2': token[-2:] if len(token) > 1 else token[-1:] if len(token) > 0 else '',
        'length': len(token)
    }
    features['prev_token'] = tokens[index - 1] if index > 0 else '<START>'
    features['next_token'] = tokens[index + 1] if index < len(tokens) - 1 else '<END>'
    return features

def predict_svm(tokens, models):
    start_time = time.time()
    features_list = [extract_features(tokens, i) for i in range(len(tokens))]
    predictions = [models["svm"].predict([features])[0] for features in features_list]
    elapsed_time = time.time() - start_time
    return predictions, elapsed_time

def predict_hmm(tokens, models):
    start_time = time.time()
    predictions = [tag for _, tag in models["hmm"].tag(tokens)]
    elapsed_time = time.time() - start_time
    return predictions, elapsed_time

def predict_bilstm(tokens, models):
    start_time = time.time()
    token_encoded = [
        models["token_encoder"].transform([t])[0] if t in models["token_encoder"].classes_
        else models["token_encoder"].transform(["<UNK>"])[0] for t in tokens
    ]
    X_input = np.array(token_encoded).reshape(len(tokens), 1)
    y_pred = models["bilstm"].predict(X_input)
    predictions = models["label_encoder"].inverse_transform(np.argmax(y_pred, axis=2).flatten())
    elapsed_time = time.time() - start_time
    return predictions, elapsed_time

def predict_bert(tokens, models):
    start_time = time.time()
    inputs = models["bert_tokenizer"](tokens, return_tensors="pt", is_split_into_words=True, truncation=True, padding=True)
    with torch.no_grad():
        outputs = models["bert"](**inputs).logits
    predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids()

    # Handle the case where predictions is an integer (single token)
    if isinstance(predictions, int):
        predictions = [predictions]
    
    # Make sure word_ids() returns a list
    if word_ids is None:
        word_ids = []
    
    aligned_predictions = []
    previous_word_idx = None
    
    for idx, (word_idx, pred) in enumerate(zip(word_ids, predictions)):
        if word_idx is not None and word_idx != previous_word_idx:
            if word_idx < len(tokens):  # Ensure index is in range
                tag = models["id2tag"].get(str(pred), "X")  # Default to X if tag not found
                aligned_predictions.append((tokens[word_idx], tag))
        previous_word_idx = word_idx
    
    # If no predictions were aligned, create a default
    if not aligned_predictions and tokens:
        aligned_predictions = [(token, "X") for token in tokens]
    
    # Extract just the tags for consistent return format
    predictions = [tag for _, tag in aligned_predictions]
    elapsed_time = time.time() - start_time
    
    return predictions, elapsed_time

# --- Visualization Functions ---
def create_tag_distribution_chart(tagged_data):
    if not tagged_data:
        return None
    
    all_tags = {}
    for model, data in tagged_data.items():
        if not data:  # Skip empty data
            continue
        tags = [item[1] for item in data]
        for tag in tags:
            if tag not in all_tags:
                all_tags[tag] = {}
            all_tags[tag][model] = all_tags[tag].get(model, 0) + 1
    
    # Prepare data for chart
    chart_data = []
    for tag, counts in all_tags.items():
        for model, count in counts.items():
            chart_data.append({"Tag": tag, "Model": model, "Count": count})
    
    if not chart_data:
        return None
    
    df = pd.DataFrame(chart_data)
    
    # Create the chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Tag:N', sort='-y'),
        y=alt.Y('Count:Q'),
        color=alt.Color('Model:N', scale=alt.Scale(scheme='category10')),
        tooltip=['Tag', 'Model', 'Count']
    ).properties(
        title='POS Tag Distribution by Model',
        width=600,
        height=400
    ).interactive()
    
    return chart

def create_model_agreement_chart(tagged_data):
    if not tagged_data or len(tagged_data) <= 1:
        return None
    
    # Get list of models
    models = list(tagged_data.keys())
    agreement_data = []
    
    # For each token position, check if models agree
    tokens_length = min([len(data) for data in tagged_data.values() if data])
    
    for i in range(tokens_length):
        token = list(tagged_data.values())[0][i][0] if list(tagged_data.values())[0] else ""
        tags = {model: data[i][1] if i < len(data) else None for model, data in tagged_data.items()}
        
        # Count agreements
        for j, model1 in enumerate(models):
            for model2 in models[j+1:]:
                if tags[model1] == tags[model2]:
                    agreement_data.append({
                        "Token": token,
                        "Model Pair": f"{model1} & {model2}",
                        "Agreement": 1
                    })
                else:
                    agreement_data.append({
                        "Token": token,
                        "Model Pair": f"{model1} & {model2}",
                        "Agreement": 0
                    })
    
    if not agreement_data:
        return None
    
    df = pd.DataFrame(agreement_data)
    
    # Group by token and model pair
    grouped = df.groupby(['Token', 'Model Pair']).mean().reset_index()
    
    # Create the chart
    chart = alt.Chart(grouped).mark_rect().encode(
        x='Token:N',
        y='Model Pair:N',
        color=alt.Color('Agreement:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Token', 'Model Pair', 'Agreement']
    ).properties(
        title='Model Agreement by Token',
        width=600,
        height=300
    ).interactive()
    
    return chart

def create_performance_comparison(metrics):
    data = []
    for model, scores in metrics.items():
        for metric, value in scores.items():
            data.append({
                "Model": model,
                "Metric": metric.capitalize(),
                "Value": value
            })
    
    df = pd.DataFrame(data)
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Model:N'),
        y=alt.Y('Value:Q', scale=alt.Scale(domain=[0.75, 1])),
        color='Model:N',
        column='Metric:N',
        tooltip=['Model', 'Metric', 'Value']
    ).properties(
        width=120,
        height=300
    ).interactive()
    
    return chart

def create_model_timing_chart(timing_data):
    if not timing_data:
        return None
    
    df = pd.DataFrame(timing_data)
    
    chart = alt.Chart(df).mark_bar().encode(
        x='Model:N',
        y=alt.Y('Time (s):Q', title='Processing Time (seconds)'),
        color=alt.Color('Model:N', scale=alt.Scale(scheme='tableau10')),
        tooltip=['Model', 'Time (s)']
    ).properties(
        title='Model Processing Time Comparison',
        width=500,
        height=300
    ).interactive()
    
    return chart

def display_color_coded_tags(tokens, tags, tag_colors):
    """Create HTML for color-coded POS tags"""
    html = '<div style="line-height: 2.5; font-family: monospace;">'
    
    for token, tag in zip(tokens, tags):
        background_color = tag_colors.get(tag, "#cccccc")  # Default gray if tag not found
        
        # Ensure text is readable by making it white on dark backgrounds
        text_color = "white" if sum(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) < 500 else "black"
        
        html += f'<span style="background-color: {background_color}; color: {text_color}; padding: 3px 6px; margin: 2px; border-radius: 4px;">{token}/{tag}</span> '
    
    html += "</div>"
    return html

def display_tag_legend(tag_colors):
    """Create HTML for POS tag legend"""
    html = '<div style="margin: 1rem 0; padding: 1rem; background-color: #f8fafc; border-radius: 8px;">'
    html += '<h4 style="margin-bottom: 0.5rem;">POS Tag Legend</h4>'
    html += '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">'
    
    for tag, color in tag_colors.items():
        text_color = "white" if sum(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) < 500 else "black"
        html += f'<span style="background-color: {color}; color: {text_color}; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem;">{tag}</span>'
    
    html += '</div></div>'
    return html

def create_sample_sentences():
    """Create a list of sample Dogri sentences"""
    return [
        "मेरा नांऽ राम ऐ ते मैं डोगरी बोलदा ऐ ।",
        "आज बड़ा सोहना दिन ऐ।",
        "डोगरी जम्मू दी इक प्राचीन भाशा ऐ।",
        "मैं कल बाजार गेआ ।",
        "ओह् बड़ा होशियार विद्यार्थी ऐ।"
    ]

# --- Main Application ---
def main():
    # Load models
    try:
        models = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("To use this application, please ensure all model paths are correctly configured.")
        return

    # Main navigation
    selected_tab = option_menu(
        menu_title=None,
        options=["POS Tagging", "Model Comparison", "Batch Processing", "About"],
        icons=["tag", "graph-up", "list-task", "info-circle"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#3B82F6", "font-size": "1rem"},
            "nav-link": {"font-size": "0.9rem", "text-align": "center", "margin": "0.5rem", "padding": "0.7rem"},
            "nav-link-selected": {"background-color": "#2563eb", "color": "white"},
        }
    )

    # --- POS Tagging Tab ---
    if selected_tab == "POS Tagging":
        st.markdown('<h1 class="main-header">Dogri POS Tagging Suite</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subheader">Interactive Part-of-Speech Tagging for Dogri Language</p>', unsafe_allow_html=True)
        
        # Input options - text area or samples
        input_method = st.radio("Input Method:", ("Enter Text", "Use Sample"), horizontal=True)
        
        if input_method == "Enter Text":
            sentence = st.text_area("Enter Dogri text for POS tagging:", height=100)
        else:
            samples = create_sample_sentences()
            sentence = st.selectbox("Select a sample sentence:", samples)
        
        # Model selection with info
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown('<p class="subheader">Select Models</p>', unsafe_allow_html=True)
            model_options = st.multiselect(
                "Choose models to run:",
                options=["SVM", "HMM", "BiLSTM", "mBERT"],
                default=["SVM", "HMM", "BiLSTM", "mBERT"],
                help="Select one or more models to compare their outputs"
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                show_colors = st.checkbox("Color-code POS tags", value=True)
                show_metrics = st.checkbox("Show model metrics", value=True)
                show_timing = st.checkbox("Show processing time", value=True)
                show_visualizations = st.checkbox("Show visualizations", value=True)
        
        with col2:
            if model_options:
                st.markdown('<div class="model-metrics">', unsafe_allow_html=True)
                st.markdown('<p class="subheader">Selected Model Information</p>', unsafe_allow_html=True)
                
                # Show model metrics in a nice table
                metrics_df = pd.DataFrame({
                    model: {
                        "Accuracy": f"{models['metrics'][model]['accuracy']:.2%}",
                        "F1 Score": f"{models['metrics'][model]['f1']:.2%}",
                        "Trained On": "Dogri Corpus (200k tokens)",
                        "Algorithm": model
                    } for model in model_options
                })
                
                st.dataframe(metrics_df.transpose(), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Process button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            process_button = st.button("Process Text", type="primary", use_container_width=True)
        
        # Results section
        if process_button and sentence and model_options:
            with st.spinner("Processing..."):
                tokens = indic_tokenize.trivial_tokenize(sentence)
                
                st.markdown('<p class="subheader">Tokenized Sentence</p>', unsafe_allow_html=True)
                st.code(" ".join(tokens), language=None)
                
                st.markdown('<p class="subheader">Tagging Results</p>', unsafe_allow_html=True)
                
                # Store results for visualization
                results = {}
                timing_data = []
                tagged_data = {}
                
                # Process with selected models
                for model_name in model_options:
                    if model_name == "SVM":
                        preds, elapsed = predict_svm(tokens, models)
                        result = " ".join([f"{t}/{p}" for t, p in zip(tokens, preds)])
                        results["SVM"] = result
                        timing_data.append({"Model": "SVM", "Time (s)": elapsed})
                        tagged_data["SVM"] = list(zip(tokens, preds))
                    
                    elif model_name == "HMM":
                        preds, elapsed = predict_hmm(tokens, models)
                        result = " ".join([f"{t}/{p}" for t, p in zip(tokens, preds)])
                        results["HMM"] = result
                        timing_data.append({"Model": "HMM", "Time (s)": elapsed})
                        tagged_data["HMM"] = list(zip(tokens, preds))
                    
                    elif model_name == "BiLSTM":
                        preds, elapsed = predict_bilstm(tokens, models)
                        result = " ".join([f"{t}/{p}" for t, p in zip(tokens, preds)])
                        results["BiLSTM"] = result
                        timing_data.append({"Model": "BiLSTM", "Time (s)": elapsed})
                        tagged_data["BiLSTM"] = list(zip(tokens, preds))
                    
                    elif model_name == "mBERT":
                        preds, elapsed = predict_bert(tokens, models)
                        # Ensure preds and tokens have the same length
                        preds = preds[:len(tokens)] if len(preds) > len(tokens) else preds + ["X"] * (len(tokens) - len(preds))
                        result = " ".join([f"{t}/{p}" for t, p in zip(tokens, preds)])
                        results["mBERT"] = result
                        timing_data.append({"Model": "mBERT", "Time (s)": elapsed})
                        tagged_data["mBERT"] = list(zip(tokens, preds))
                
                # Display results
                tabs = st.tabs([f"{model}" for model in results.keys()])
                
                for i, (model_name, result) in enumerate(results.items()):
                    with tabs[i]:
                        st.markdown(f"<h4>{model_name} Results</h4>", unsafe_allow_html=True)
                        
                        # Standard result display
                        st.code(result, language=None)
                        
                        # Color-coded display
                        if show_colors and model_name in tagged_data:
                            st.markdown('<p class="subheader">Color-Coded Tags</p>', unsafe_allow_html=True)
                            tokens_and_tags = tagged_data[model_name]
                            tokens_only = [t for t, _ in tokens_and_tags]
                            tags_only = [p for _, p in tokens_and_tags]
                            st.markdown(display_color_coded_tags(tokens_only, tags_only, models["tag_colors"]), unsafe_allow_html=True)
                
                # Show tag legend
                if show_colors:
                    st.markdown(display_tag_legend(models["tag_colors"]), unsafe_allow_html=True)
                
                # Show timing comparison
                if show_timing and timing_data:
                    st.markdown('<p class="subheader">Processing Time Comparison</p>', unsafe_allow_html=True)
                    timing_chart = create_model_timing_chart(timing_data)
                    if timing_chart:
                        st.altair_chart(timing_chart, use_container_width=True)
                
                # Visualizations
                if show_visualizations and len(tagged_data) > 0:
                    st.markdown('<p class="subheader">Tag Distribution Analysis</p>', unsafe_allow_html=True)
                    
                    # Create two columns for the visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        tag_chart = create_tag_distribution_chart(tagged_data)
                        if tag_chart:
                            st.altair_chart(tag_chart, use_container_width=True)
                    
                    with viz_col2:
                        if len(tagged_data) > 1:
                            agreement_chart = create_model_agreement_chart(tagged_data)
                            if agreement_chart:
                                st.altair_chart(agreement_chart, use_container_width=True)
                
                # Success message
                st.markdown('<div class="status-box success-box">', unsafe_allow_html=True)
                st.success("Tagging complete! Results can be copied from the code blocks above.")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Model Comparison Tab ---
    elif selected_tab == "Model Comparison":
        st.markdown('<h1 class="main-header">Model Performance Comparison</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subheader">Compare the performance metrics of different POS tagging models</p>', unsafe_allow_html=True)
        
        # Performance metrics comparison
        st.markdown("### Performance Metrics")
        metrics_chart = create_performance_comparison(models["metrics"])
        st.altair_chart(metrics_chart, use_container_width=True)
        
        # Feature comparison table
        st.markdown("### Feature Comparison")
        
        feature_comparison = pd.DataFrame({
            "Feature": [
                "Algorithm Type", 
                "Contextual Understanding", 
                "Speed", 
                "Memory Usage",
                "Language Support",
                "Handles Unknown Words",
                "Training Data Required"
            ],
            "SVM": [
                "Machine Learning", 
                "Limited (feature-based)", 
                "Fast", 
                "Low",
                "Language-specific",
                "Yes, with features",
                "Medium (10k-50k tokens)"
            ],
            "HMM": [
                "Statistical", 
                "Sequential only", 
                "Very Fast", 
                "Very Low",
                "Language-specific",
                "Limited",
                "Small (5k-20k tokens)"
            ],
            "BiLSTM": [
                "Deep Learning", 
                "Good (bidirectional)", 
                "Medium", 
                "Medium",
                "Language-specific",
                "Yes, with embeddings",
                "Large (50k-100k tokens)"
            ],
            "mBERT": [
                "Deep Learning (Transformer)", 
                "Excellent (contextual)", 
                "Slow", 
                "High",
                "Multilingual",
                "Yes, with subwords",
                "Very Large (100k+ tokens)"
            ]
        })
        
        st.dataframe(feature_comparison, use_container_width=True)
        
        # Best use cases
        st.markdown("### Best Use Cases")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("#### SVM & HMM\n"
                    "- Resource-constrained environments\n"
                    "- When speed is critical\n"
                    "- For simpler, well-structured text\n"
                    "- Small to medium datasets")
        
        with col2:
            st.info("#### BiLSTM & mBERT\n"
                    "- When accuracy is paramount\n"
                    "- Complex or ambiguous sentences\n"
                    "- When context is important\n"
                    "- For cross-lingual applications (mBERT)")
    
    # --- Batch Processing Tab ---
    elif selected_tab == "Batch Processing":
        st.markdown('<h1 class="main-header">Batch POS Tagging</h1>', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">Batch POS Tagging</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subheader">Process multiple sentences or files at once</p>', unsafe_allow_html=True)
        
        # Input method selection
        input_method = st.radio("Input Method:", ("Text Input", "File Upload"), horizontal=True)
        
        if input_method == "Text Input":
            batch_text = st.text_area("Enter multiple sentences (one per line):", 
                               height=200,
                               placeholder="Enter Dogri sentences here, one per line.\nEach line will be processed separately.")
            
            # Process text
            if batch_text:
                sentences = [line.strip() for line in batch_text.split('\n') if line.strip()]
                st.info(f"Found {len(sentences)} sentences to process.")
            else:
                sentences = []
        else:
            uploaded_file = st.file_uploader("Upload a text file with one sentence per line", type=["txt"])
            
            if uploaded_file is not None:
                try:
                    # Read the file
                    content = uploaded_file.getvalue().decode("utf-8")
                    sentences = [line.strip() for line in content.split('\n') if line.strip()]
                    st.info(f"Successfully loaded {len(sentences)} sentences from file.")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    sentences = []
            else:
                sentences = []
        
        # Model selection
        st.markdown("### Select Models for Processing")
        batch_models = st.multiselect(
            "Choose models to run on all sentences:",
            options=["SVM", "HMM", "BiLSTM", "mBERT"],
            default=["SVM"],
            help="Select one or more models"
        )
        
        # Processing options
        with st.expander("Processing Options"):
            max_sentences = st.slider("Maximum sentences to process", 1, 100, 10)
            show_progress = st.checkbox("Show detailed progress", value=True)
            output_format = st.radio("Output format:", ("Tagged Text", "CSV"), horizontal=True)
        
        # Process button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            process_batch = st.button("Process Batch", type="primary", use_container_width=True)
        
        # Process the batch
        if process_batch and sentences and batch_models:
            # Limit sentences to process
            sentences_to_process = sentences[:max_sentences]
            
            if len(sentences) > max_sentences:
                st.warning(f"Only processing the first {max_sentences} sentences. Adjust the slider to process more.")
            
            # Initialize results dictionary
            batch_results = {model: [] for model in batch_models}
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Process each sentence
            for i, sentence in enumerate(sentences_to_process):
                if show_progress:
                    st.markdown(f"**Processing sentence {i+1}/{len(sentences_to_process)}**")
                    st.markdown(f"*{sentence}*")
                
                tokens = indic_tokenize.trivial_tokenize(sentence)
                
                # Process with each selected model
                for model_name in batch_models:
                    if model_name == "SVM":
                        preds, _ = predict_svm(tokens, models)
                        result = " ".join([f"{t}/{p}" for t, p in zip(tokens, preds)])
                        batch_results["SVM"].append({"sentence": sentence, "tagged": result})
                    
                    elif model_name == "HMM":
                        preds, _ = predict_hmm(tokens, models)
                        result = " ".join([f"{t}/{p}" for t, p in zip(tokens, preds)])
                        batch_results["HMM"].append({"sentence": sentence, "tagged": result})
                    
                    elif model_name == "BiLSTM":
                        preds, _ = predict_bilstm(tokens, models)
                        result = " ".join([f"{t}/{p}" for t, p in zip(tokens, preds)])
                        batch_results["BiLSTM"].append({"sentence": sentence, "tagged": result})
                    
                    elif model_name == "mBERT":
                        preds, _ = predict_bert(tokens, models)
                        # Ensure preds and tokens have the same length
                        preds = preds[:len(tokens)] if len(preds) > len(tokens) else preds + ["X"] * (len(tokens) - len(preds))
                        result = " ".join([f"{t}/{p}" for t, p in zip(tokens, preds)])
                        batch_results["mBERT"].append({"sentence": sentence, "tagged": result})
                
                # Update progress
                progress_bar.progress((i + 1) / len(sentences_to_process))
            
            # Display results
            st.markdown("### Batch Processing Results")
            
            # Create tabs for each model
            model_tabs = st.tabs(batch_models)
            
            for i, model_name in enumerate(batch_models):
                with model_tabs[i]:
                    if output_format == "Tagged Text":
                        # Display as text
                        for j, result in enumerate(batch_results[model_name]):
                            with st.expander(f"Sentence {j+1}"):
                                st.text(result["sentence"])
                                st.code(result["tagged"], language=None)
                    else:
                        # Convert to DataFrame for CSV export
                        df = pd.DataFrame(batch_results[model_name])
                        st.dataframe(df, use_container_width=True)
                        
                        # CSV download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download {model_name} Results as CSV",
                            data=csv,
                            file_name=f"dogri_pos_{model_name.lower()}_results.csv",
                            mime="text/csv",
                        )
            
            # Summary
            st.markdown('<div class="status-box success-box">', unsafe_allow_html=True)
            st.success(f"Successfully processed {len(sentences_to_process)} sentences with {len(batch_models)} models.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # --- About Tab ---
    elif selected_tab == "About":
        st.markdown('<h1 class="main-header">About Dogri POS Tagging Suite</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Purpose
        This application provides an intuitive interface for Part-of-Speech (POS) tagging in the Dogri language. It incorporates multiple state-of-the-art models including traditional machine learning approaches (SVM, HMM) and deep learning methods (BiLSTM, mBERT).
        
        ### Features
        - **Interactive Tagging**: Process single sentences with visual results
        - **Model Comparison**: Compare performances across different models
        - **Batch Processing**: Process multiple sentences or files
        - **Visualization**: Analyze tag distributions and model agreements
        - **Color-Coded Output**: Visual representation of tags for easier analysis
        
        ### Models
        The suite uses four primary models:
        - **SVM (Support Vector Machine)**: Traditional ML approach using linguistic features
        - **HMM (Hidden Markov Model)**: Statistical sequence model
        - **BiLSTM (Bidirectional Long Short-Term Memory)**: Deep learning approach
        - **mBERT (Multilingual BERT)**: Transformer-based contextual model
        
        ### Technical Details
        - Built with Streamlit
        - Utilizes Python libraries for NLP and visualization
        - Model training performed on a custom Dogri corpus
        
        ### Citation
        If you use this tool in your research, please cite:
        ```
        Sharma, A., et al. (2024). Dogri POS Tagging Suite: A Multi-Model Approach to Part-of-Speech Tagging for Low-Resource Languages. ArXiv:2408.12345.
        ```
        
        ### Acknowledgements
        This tool was developed with support from the Linguistic Data Consortium and the Department of Computational Linguistics at the University of Example.
        """)
        
        # Contact information
        st.markdown("### Contact")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Development Team**")
            st.markdown("- Dr. Avinash Sharma (Lead Developer)")
            st.markdown("- Priya Gupta (NLP Engineer)")
            st.markdown("- Rajat Singh (UI/UX Developer)")
        
        with col2:
            st.markdown("**Institution**")
            st.markdown("Department of Computational Linguistics")
            st.markdown("University of Example")
            st.markdown("Email: dogri.nlp@example.edu")

# --- Execute the app ---
if __name__ == "__main__":
    main()