import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog
import base64
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background-color: #0b1121;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.1) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(14, 165, 233, 0.1) 0%, transparent 40%);
        color: #f8fafc;
    }

    .hero-container {
        text-align: center;
        padding: 4rem 0 2rem 0;
        animation: fadeInDown 1s ease-out;
    }
    .hero-title {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #34d399 0%, #0ea5e9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -2px;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        margin-top: 1rem;
        font-weight: 400;
    }

    /* MASSIVE BUTTONS FOR UPLOAD REPLACEMENT */
    div[data-testid="stButton"] button {
        height: 250px !important;
        width: 100% !important;
        border-radius: 24px !important;
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease !important;
        animation: fadeInUp 1s ease-out 0.2s both;
    }

    div[data-testid="stButton"] button p {
        white-space: pre-wrap !important;
        font-size: 1.6rem !important;
        color: #f8fafc !important;
        text-align: center;
        line-height: 1.6;
    }

    div[data-testid="stButton"] button:hover {
        border-color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.1) !important;
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    }

    div[data-testid="stButton"] button:hover p {
        color: #38bdf8 !important;
    }

    /* Results Card - Single */
    .result-card {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 20px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        animation: scaleIn 0.5s ease-out;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    /* Results Card - Grid */
    .folder-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
        gap: 20px;
        padding: 10px;
        margin-bottom: 2rem;
    }
    .icon-card {
        background: rgba(15, 23, 42, 0.4);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        transition: transform 0.2s, background 0.2s, z-index 0s;
        position: relative;
        z-index: 1;
    }
    .icon-card:hover {
        transform: translateY(-3px);
        background: rgba(15, 23, 42, 0.8);
        border-color: rgba(255, 255, 255, 0.15);
        z-index: 100;
    }
    .icon-thumbnail {
        width: 110px;
        height: 110px;
        border-radius: 8px;
        cursor: pointer;
        object-fit: cover;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 8px;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease, z-index 0s;
        position: relative;
        z-index: 1;
    }
    .icon-thumbnail:hover {
        transform: scale(2.5);
        z-index: 100;
        box-shadow: 0 15px 35px rgba(0,0,0,0.6);
    }
    .lightbox-thumbnail {
        width: 100%;
        height: auto;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        object-fit: cover;
        transition: transform 0.3s ease;
    }
    .lightbox-thumbnail:hover {
        transform: scale(1.02);
    }
    .icon-filename {
        font-size: 0.75rem;
        color: #cbd5e1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
        margin-bottom: 4px;
    }
    .icon-value-bio {
        font-size: 0.85rem;
        font-weight: 800;
        color: #10b981;
    }
    .icon-value-nonbio {
        font-size: 0.85rem;
        font-weight: 800;
        color: #ef4444;
    }

    .result-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }

    .result-value-bio {
        font-size: 3rem;
        font-weight: 800;
        color: #10b981;
        margin: 0;
        text-shadow: 0 0 30px rgba(16, 185, 129, 0.4);
        line-height: 1.1;
    }

    .result-value-nonbio {
        font-size: 3rem;
        font-weight: 800;
        color: #ef4444;
        margin: 0;
        text-shadow: 0 0 30px rgba(239, 68, 68, 0.4);
        line-height: 1.1;
    }
    
    .grid-value-bio {
        font-size: 1.8rem;
        font-weight: 800;
        color: #10b981;
        margin: 0.5rem 0 0.5rem 0;
        text-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }

    .grid-value-nonbio {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ef4444;
        margin: 0.5rem 0 0.5rem 0;
        text-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }

    .confidence-wrapper { margin-top: 2.5rem; width: 100%; }
    .grid-confidence-wrapper { margin-top: 1rem; width: 100%; }

    .confidence-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        font-size: 1rem;
        color: #cbd5e1;
    }
    
    .grid-confidence-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #cbd5e1;
    }

    .confidence-track {
        height: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        overflow: hidden;
        width: 100%;
    }

    .badge-bio {
        display: inline-block;
        padding: 0.8rem 1.8rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        border-radius: 999px;
        font-weight: 600;
        margin-top: 2.5rem;
        font-size: 1.1rem;
    }

    .badge-nonbio {
        display: inline-block;
        padding: 0.8rem 1.8rem;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
        border-radius: 999px;
        font-weight: 600;
        margin-top: 2.5rem;
        font-size: 1.1rem;
    }

    @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes scaleIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def get_base64_of_image(image):
    buffered = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=70, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()

# --- LOAD MODEL CACHING ---
@st.cache_resource(show_spinner=False)
def load_classification_model():
    model_path = "garbage_classifier_model.h5"
    if os.path.exists(model_path):
        tf.keras.backend.clear_session() # Clear session to prevent threading issues
        return tf.keras.models.load_model(model_path)
    else:
        return None

# --- INFERENCE FUNCTION ---
def predict_image(model, image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Exact preprocessing match for Keras ImageDataGenerator
    img_resized = image.resize((224, 224))
    input_img = np.array(img_resized) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    prediction = model.predict(input_img, verbose=0)[0][0]
    
    if prediction < 0.5:
        return "Biodegradable", (1 - prediction) * 100, True
    else:
        return "Non-Biodegradable", prediction * 100, False

# --- MAIN APP LAYOUT ---
st.markdown('''
<div class="hero-container">
    <h1 class="hero-title">EcoVision AI</h1>
    <p class="hero-subtitle">Next-Generation Waste Classification powered by Deep Learning</p>
</div>
''', unsafe_allow_html=True)

with st.spinner("Initializing AI Core..."):
    model = load_classification_model()

if model is None:
    st.error("Neural Network not found. Please ensure 'garbage_classifier_model.h5' is available.")
else:
    # Use native OS dialogs via Tkinter bound to large buttons
    up_col1, up_col2 = st.columns(2, gap="large")
    
    with up_col1:
        if st.button("📁\n\nOpen a Folder\n\nClick to select an entire directory", use_container_width=True):
            import subprocess
            script = "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.attributes('-topmost', True); root.withdraw(); print(filedialog.askdirectory(master=root, title='Select Folder for Batch Analysis')); root.destroy()"
            result = subprocess.run(["python", "-c", script], capture_output=True, text=True)
            folder_path = result.stdout.strip()
            if folder_path:
                st.session_state.selected_folder = folder_path
                st.session_state.selected_file = None
                
    with up_col2:
        if st.button("🖼️\n\nChoose an Image\n\nClick to select a single image file", use_container_width=True):
            import subprocess
            script = "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.attributes('-topmost', True); root.withdraw(); print(filedialog.askopenfilename(master=root, title='Select Image File', filetypes=[('Image files', '*.jpg *.jpeg *.png *.webp')])); root.destroy()"
            result = subprocess.run(["python", "-c", script], capture_output=True, text=True)
            file_path = result.stdout.strip()
            if file_path:
                st.session_state.selected_file = file_path
                st.session_state.selected_folder = None
                
    folder_path = st.session_state.get('selected_folder')
    file_path = st.session_state.get('selected_file')
    
    if folder_path:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        valid_exts = ['.jpg', '.jpeg', '.png', '.webp']
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
        
        if len(files) == 0:
            st.warning("No valid images found in the selected folder.")
        else:
            st.markdown(f"<h3 style='text-align:center; color:#f8fafc;'>Batch Analysis Results ({len(files)} items)</h3><br>", unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            folder_html = '<div class="folder-grid">'
            for idx, filepath in enumerate(files):
                progress_bar.progress((idx + 1) / len(files))
                status_text.text(f"Processing image {idx + 1} of {len(files)}...")
                
                image = Image.open(filepath)
                
                # Resize image to prevent massive memory usage and browser lag, but keep it clear enough for full-screen cover
                image.thumbnail((800, 800))
                
                b64_img = get_base64_of_image(image)
                
                label, confidence, is_biodegradable = predict_image(model, image)
                res_class = "icon-value-bio" if is_biodegradable else "icon-value-nonbio"
                bar_color = "#10b981" if is_biodegradable else "#ef4444"
                filename = os.path.basename(filepath)
                
                lb_id = f"lb_{idx}"
                
                folder_html += f"""
<div class="icon-card">
<img src="data:image/jpeg;base64,{b64_img}" class="icon-thumbnail" alt="Thumbnail">
<div class="icon-filename" title="{filename}">{filename}</div>
<div class="{res_class}">{label}</div>
<div style="font-size: 0.75rem; color:{bar_color}; font-weight:600; margin-top:2px;">{confidence:.0f}%</div>
</div>
"""
            folder_html += '</div>'
            
            progress_bar.empty()
            status_text.empty()
            
            st.markdown(folder_html, unsafe_allow_html=True)

    elif file_path:
        st.markdown("<br><br>", unsafe_allow_html=True)
        image = Image.open(file_path)
        image.thumbnail((800, 800))
        
        res_col_left, res_col_main, res_col_right = st.columns([1, 4, 1])
        with res_col_main:
            img_col, info_col = st.columns([1, 1], gap="large")
            with img_col:
                b64_single = get_base64_of_image(image)
                lb_id = "lb_single"
                single_html = f"""
<div style="width:100%;">
<img src="data:image/jpeg;base64,{b64_single}" class="lightbox-thumbnail" style="width:100%;">
</div>
"""
                st.markdown(single_html, unsafe_allow_html=True)
                
            with info_col:
                with st.spinner("Processing visual data..."):
                    label, confidence, is_biodegradable = predict_image(model, image)
                
                res_class = "result-value-bio" if is_biodegradable else "result-value-nonbio"
                bar_color = "#10b981" if is_biodegradable else "#ef4444"
                badge_class = "badge-bio" if is_biodegradable else "badge-nonbio"
                badge_icon = "🌿 Safe for Composting" if is_biodegradable else "⚠️ Requires Special Disposal"
                
                result_html = f"""
<div class="result-card">
<div class="result-label">Classification Output</div>
<div class="{res_class}">{label}</div>
<div class="confidence-wrapper">
<div class="confidence-label">
<span>Confidence Score</span>
<span style="font-weight: 600; color: {bar_color};">{confidence:.1f}%</span>
</div>
<div class="confidence-track">
<div style="width: {confidence}%; height: 100%; background: {bar_color}; border-radius: 5px; box-shadow: 0 0 15px {bar_color}; animation: slideRight 1.5s ease-out;"></div>
</div>
</div>
<div>
<div class="{badge_class}">
{badge_icon}
</div>
</div>
</div>
<style>
@keyframes slideRight {{
from {{ width: 0%; }}
to {{ width: {confidence}%; }}
}}
</style>
"""
                st.markdown(result_html, unsafe_allow_html=True)
