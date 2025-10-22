"""Streamlit application for breast cancer detection."""
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = f"http://{config.API_HOST}:{config.API_PORT}"


def check_api_health():
    """Check if API is running and healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None


def segment_image(image_file):
    """Send image to API for segmentation."""
    try:
        files = {"file": image_file}
        response = requests.post(f"{API_URL}/predict/segment", files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def visualize_segmentation(image, mask, unique_labels):
    """Visualize original image and segmentation mask."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(mask, cmap='viridis')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image, cmap='gray', alpha=0.7)
    axes[2].imshow(mask, cmap='viridis', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🏥 Breast Cancer Detection System")
    st.markdown("""
    This application uses deep learning to analyze thermal breast images for:
    - **Segmentation**: Identifying breast regions
    - **Classification**: Detecting potential abnormalities (coming soon)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This system combines:
        - **R2AU-Net** for breast segmentation
        - **VICReg** pretrained backbone for classification
        - Self-supervised learning techniques
        """)
        
        st.header("API Status")
        is_healthy, health_data = check_api_health()
        
        if is_healthy:
            st.success("✅ API is running")
            if health_data and 'models_loaded' in health_data:
                st.write("**Models Loaded:**")
                for model_name, loaded in health_data['models_loaded'].items():
                    status = "✅" if loaded else "❌"
                    st.write(f"{status} {model_name.capitalize()}")
        else:
            st.error("❌ API is not responding")
            st.warning(f"Make sure the API is running at {API_URL}")
    
    # Main content
    tab1, tab2 = st.tabs(["📤 Upload & Analyze", "ℹ️ Information"])
    
    with tab1:
        st.header("Upload Thermal Breast Image")
        
        uploaded_file = st.file_uploader(
            "Choose a thermal image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a thermal breast image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Image Information")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Mode:** {image.mode}")
            
            # Analysis buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔍 Run Segmentation", type="primary", use_container_width=True):
                    if not is_healthy:
                        st.error("API is not running. Please start the API first.")
                    else:
                        with st.spinner("Running segmentation..."):
                            # Reset file pointer
                            uploaded_file.seek(0)
                            result = segment_image(uploaded_file)
                            
                            if result and result.get('success'):
                                st.success("Segmentation completed!")
                                
                                # Extract results
                                prediction = result.get('prediction', {})
                                mask = np.array(prediction.get('segmentation_mask'))
                                unique_labels = prediction.get('unique_labels', [])
                                
                                # Store in session state
                                st.session_state['mask'] = mask
                                st.session_state['unique_labels'] = unique_labels
            
            with col2:
                if st.button("🧬 Run Classification", type="secondary", use_container_width=True, disabled=True):
                    st.info("Classification feature coming soon!")
            
            with col3:
                if st.button("🔬 Full Analysis", type="secondary", use_container_width=True, disabled=True):
                    st.info("Full analysis feature coming soon!")
            
            # Display results if available
            if 'mask' in st.session_state:
                st.markdown("---")
                st.header("Segmentation Results")
                
                mask = st.session_state['mask']
                unique_labels = st.session_state['unique_labels']
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Regions", len(unique_labels))
                with col2:
                    st.metric("Image Size", f"{mask.shape[0]}x{mask.shape[1]}")
                with col3:
                    breast_pixels = np.sum(mask > 0)
                    total_pixels = mask.size
                    percentage = (breast_pixels / total_pixels) * 100
                    st.metric("Breast Coverage", f"{percentage:.1f}%")
                
                # Visualize
                st.subheader("Visualization")
                fig = visualize_segmentation(np.array(image.convert('L')), mask, unique_labels)
                st.pyplot(fig)
                
                # Show label information
                with st.expander("Label Information"):
                    st.write("**Detected Labels:**")
                    label_names = {0: "Background", 1: "Left Breast", 2: "Right Breast"}
                    for label in unique_labels:
                        label_name = label_names.get(label, f"Unknown ({label})")
                        pixel_count = np.sum(mask == label)
                        st.write(f"- {label_name}: {pixel_count:,} pixels")
    
    with tab2:
        st.header("About the Project")
        
        st.markdown("""
        ### Breast Cancer Detection using Thermal Imaging
        
        This project leverages **artificial intelligence** and **thermal imaging** to assist in 
        early breast cancer detection. The system uses a two-stage approach:
        
        #### 1. Segmentation Stage
        - **Model**: R2 Attention U-Net (R2AU-Net)
        - **Purpose**: Identifies and segments breast regions from thermal images
        - **Technology**: Recurrent residual convolutional blocks with attention mechanisms
        
        #### 2. Classification Stage (Coming Soon)
        - **Model**: VICReg-pretrained ResNet50
        - **Purpose**: Classifies images for potential cancer indicators
        - **Technology**: Self-supervised learning with fine-tuning
        
        ### Key Features
        - ✅ **Non-invasive**: Uses thermal imaging instead of radiation
        - ✅ **AI-powered**: Deep learning models trained on medical data
        - ✅ **Accessible**: Web-based interface for easy use
        - ✅ **Research-backed**: Based on published scientific methods
        
        ### Technical Details
        - **Framework**: PyTorch for model development
        - **Backend**: FastAPI for RESTful API
        - **Frontend**: Streamlit for web interface
        - **Training**: Weights & Biases for experiment tracking
        
        ### Disclaimer
        This is a research tool and should not be used as a substitute for professional medical advice, 
        diagnosis, or treatment. Always consult with qualified healthcare providers.
        """)
        
        st.markdown("---")
        st.markdown("""
        ### References
        - R2AU-Net: [Image Segmentation by LeeJunHyun](https://github.com/LeeJunHyun/Image_Segmentation)
        - VICReg: [Meta AI Research](https://github.com/facebookresearch/vicreg)
        - DMR-IR Dataset: Public thermal breast imaging dataset
        """)


if __name__ == "__main__":
    main()
