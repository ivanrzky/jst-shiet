"""
Streamlit Web UI for Handwritten Letter Recognition (A-J)
"""

import os
# Set environment variables before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf

from model import (
    load_and_preprocess_data, 
    build_cnn_model, 
    train_model, 
    predict_single, 
    evaluate_model,
    get_data_info,
    ALPHABET
)

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Handwritten Letter Recognition (A-J)",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-correct {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-wrong {
        color: #dc3545;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ================== STREAMLIT CALLBACK ==================
class StreamlitCallback(tf.keras.callbacks.Callback):
    """Custom callback for Streamlit progress updates during training."""
    
    def __init__(self, epochs, progress_bar, status_text):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        
    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs} - Accuracy: {acc:.4f} - Val Accuracy: {val_acc:.4f}")


# ================== DATA CACHING ==================
@st.cache_data(show_spinner=False)
def get_data():
    """Load and cache the preprocessed data."""
    return load_and_preprocess_data("A_Z Handwritten Data.csv")


# ================== SESSION STATE INIT ==================
def init_session_state():
    """Initialize session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'test_acc' not in st.session_state:
        st.session_state.test_acc = None
    if 'test_loss' not in st.session_state:
        st.session_state.test_loss = None
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'trained_epochs' not in st.session_state:
        st.session_state.trained_epochs = 0


# ================== MAIN APP ==================
def main():
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">✍️ Handwritten Letter Recognition</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">CNN-based recognition for letters A-J using TensorFlow</p>', unsafe_allow_html=True)
    
    # Check if dataset exists
    if not os.path.exists("A_Z Handwritten Data.csv"):
        st.error("❌ Dataset tidak ditemukan! Pastikan file 'A_Z Handwritten Data.csv' ada di folder yang sama.")
        st.stop()
    
    # Load data
    with st.spinner("Loading dataset..."):
        try:
            x_train, x_test, y_train, y_test, labels_test = get_data()
            data_info = get_data_info(x_train, x_test, labels_test)
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.stop()
    
    # ================== SIDEBAR ==================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset Info
        st.subheader("📊 Dataset Info")
        st.info(f"""
        - **Training samples:** {data_info['train_samples']:,}
        - **Test samples:** {data_info['test_samples']:,}
        - **Image size:** 28 × 28
        - **Classes:** A-J (10 classes)
        """)
        
        st.divider()
        
        # Training Section
        st.subheader("🏋️ Training Configuration")
        epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=10, step=1)
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)
        
        train_button = st.button("🚀 Train Model", use_container_width=True, type="primary")
        
        st.divider()
        
        # Prediction Section
        st.subheader("🔮 Prediction")
        max_idx = len(x_test) - 1
        
        # Initialize test_idx in session state
        if 'test_idx' not in st.session_state:
            st.session_state.test_idx = 0
        
        # Random index button
        if st.button("🎲 Random Index", use_container_width=True):
            st.session_state.test_idx = np.random.randint(0, max_idx + 1)
        
        test_idx = st.number_input(
            "Test Image Index", 
            min_value=0, 
            max_value=max_idx, 
            value=st.session_state.test_idx, 
            step=1,
            key="test_idx",
            help=f"Select an index between 0 and {max_idx}"
        )
        
        predict_button = st.button("🔍 Predict", use_container_width=True, type="secondary")
        
        st.divider()
        
        # Model Status
        st.subheader("📈 Model Status")
        if st.session_state.model is not None:
            st.success(f"✅ Model trained ({st.session_state.trained_epochs} epochs)")
            if st.session_state.test_acc is not None:
                st.metric("Test Accuracy", f"{st.session_state.test_acc:.2%}")
        else:
            st.warning("⚠️ Model belum di-train")
    
    # ================== MAIN CONTENT ==================
    
    # Training Logic
    if train_button:
        st.subheader("🏋️ Training Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Building model..."):
            model = build_cnn_model()
        
        status_text.text("Starting training...")
        
        # Custom callback for progress
        callback = StreamlitCallback(epochs, progress_bar, status_text)
        
        # Train
        history = train_model(
            model, x_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[callback]
        )
        
        # Evaluate
        test_loss, test_acc = evaluate_model(model, x_test, y_test)
        
        # Save to session state
        st.session_state.model = model
        st.session_state.history = history
        st.session_state.test_acc = test_acc
        st.session_state.test_loss = test_loss
        st.session_state.trained_epochs = epochs
        
        progress_bar.progress(1.0)
        status_text.text("Training completed!")
        
        st.success(f"✅ Training selesai! Test Accuracy: {test_acc:.2%}")
        # st.balloons()
    
    # ================== RESULTS DISPLAY ==================
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🖼️ Test Image")
        
        if predict_button or st.session_state.last_prediction is not None:
            if st.session_state.model is None:
                st.warning("⚠️ Train model terlebih dahulu sebelum melakukan prediksi!")
            else:
                # Make prediction
                if predict_button:
                    prediction = predict_single(
                        st.session_state.model, 
                        x_test, 
                        labels_test, 
                        test_idx
                    )
                    st.session_state.last_prediction = prediction
                    st.session_state.last_idx = test_idx
                else:
                    prediction = st.session_state.last_prediction
                    test_idx = st.session_state.get('last_idx', 0)
                
                # Display image
                img = prediction['image'].reshape(28, 28)
                st.image(img, width=200, caption=f"Test Index: {test_idx}", clamp=True)
                
                # Prediction results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                true_label = prediction['true_label']
                pred_label = prediction['predicted_label']
                confidence = prediction['confidence']
                
                is_correct = true_label == pred_label
                
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    st.metric("True Label", true_label)
                with result_col2:
                    if is_correct:
                        st.metric("Predicted", pred_label, delta="Correct ✓", delta_color="normal")
                    else:
                        st.metric("Predicted", pred_label, delta="Wrong ✗", delta_color="inverse")
                
                # Confidence
                st.metric("Confidence Score", f"{confidence:.2%}")
                
                # Progress bar for confidence
                st.progress(confidence)
                
        else:
            st.info("👆 Pilih index dan klik 'Predict' untuk melihat hasil prediksi")
            # Show sample image
            img = x_test[0].reshape(28, 28)
            st.image(img, width=200, caption="Sample Image (Index: 0)", clamp=True)
    
    with col2:
        # Confidence Bar Chart
        if st.session_state.last_prediction is not None:
            st.subheader("📊 Confidence Scores (All Classes)")
            
            prediction = st.session_state.last_prediction
            confidence_scores = prediction['confidence_scores']
            
            # Create bar chart
            df_confidence = pd.DataFrame({
                'Letter': list(confidence_scores.keys()),
                'Confidence': list(confidence_scores.values())
            })
            
            # Color bars based on predicted vs true label
            colors = []
            for letter in df_confidence['Letter']:
                if letter == prediction['predicted_label']:
                    colors.append('#667eea')  # Purple for predicted
                elif letter == prediction['true_label'] and prediction['true_label'] != prediction['predicted_label']:
                    colors.append('#28a745')  # Green for true (if different)
                else:
                    colors.append('#dee2e6')  # Gray for others
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df_confidence['Letter'],
                    y=df_confidence['Confidence'],
                    marker_color=colors,
                    text=[f"{v:.1%}" for v in df_confidence['Confidence']],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                xaxis_title="Letter Class",
                yaxis_title="Confidence",
                yaxis_tickformat='.0%',
                yaxis_range=[0, 1.1],
                height=350,
                margin=dict(t=30, b=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("📊 Confidence Scores")
            st.info("Confidence chart akan ditampilkan setelah melakukan prediksi")
    
    # ================== TRAINING HISTORY ==================
    st.divider()
    
    if st.session_state.history is not None:
        st.subheader("📈 Training History")
        
        history = st.session_state.history.history
        epochs_range = list(range(1, len(history['accuracy']) + 1))
        
        # Create subplot with 2 charts
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Accuracy per Epoch', 'Loss per Epoch')
        )
        
        # Accuracy chart
        fig.add_trace(
            go.Scatter(x=epochs_range, y=history['accuracy'], name='Train Accuracy', 
                      line=dict(color='#667eea', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs_range, y=history['val_accuracy'], name='Val Accuracy',
                      line=dict(color='#764ba2', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Loss chart
        fig.add_trace(
            go.Scatter(x=epochs_range, y=history['loss'], name='Train Loss',
                      line=dict(color='#e74c3c', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs_range, y=history['val_loss'], name='Val Loss',
                      line=dict(color='#c0392b', width=2, dash='dash')),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Final metrics
        st.subheader("📊 Final Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Final Train Accuracy", f"{history['accuracy'][-1]:.2%}")
        with metric_col2:
            st.metric("Final Val Accuracy", f"{history['val_accuracy'][-1]:.2%}")
        with metric_col3:
            st.metric("Test Accuracy", f"{st.session_state.test_acc:.2%}")
        with metric_col4:
            st.metric("Test Loss", f"{st.session_state.test_loss:.4f}")
    else:
        st.subheader("📈 Training History")
        st.info("📊 Training history akan ditampilkan setelah model di-train")
    
    # ================== FOOTER ==================
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Handwritten Letter Recognition (A-J) | CNN with TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
