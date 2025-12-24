# streamlit_app.py - Deploy this on Streamlit Cloud
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64
import urllib.request
import json

# Set page configuration
st.set_page_config(
    page_title="Composite Material Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
    }
    .prediction-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🏗️ Composite Material Tensile Strength Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
Predict tensile strength of composite materials using machine learning models trained on composition data.
Upload your data or use the interactive controls below.
""")

# Load models from GitHub (or local cache)
@st.cache_resource
def load_models_from_github():
    """Load models from GitHub repository"""
    try:
        # GitHub raw URLs (update with your actual repo URLs)
        base_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/"
        
        # Load the combined artifacts file
        try:
            # Try loading from GitHub first
            artifacts_url = base_url + "model_artifacts.pkl"
            with urllib.request.urlopen(artifacts_url) as response:
                artifacts = joblib.load(response)
                st.success("✅ Models loaded successfully from GitHub!")
        except:
            # Fallback to local file (for development)
            artifacts = joblib.load('model_artifacts.pkl')
            st.info("📁 Using local model files")
        
        return artifacts
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("""
        If models aren't trained yet:
        1. Run `train_model.py` in Google Colab
        2. Upload the model files to GitHub
        3. Update the GitHub URLs in this app
        """)
        return None

# Initialize models
artifacts = load_models_from_github()

if artifacts:
    scaler = artifacts['scaler']
    lr_model = artifacts['lr_model']
    nn_model = artifacts['nn_model']
    feature_names = artifacts['feature_names']
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Navigation")
        app_mode = st.radio(
            "Choose Mode:",
            ["🏠 Home", "📊 Manual Input", "📁 File Upload", "📈 Model Analysis", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### Model Selection")
        selected_model = st.selectbox(
            "Choose prediction model:",
            ["Linear Regression", "Neural Network", "Both (Compare)"],
            index=2
        )
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("🔄 Reset All Inputs"):
            st.rerun()
        
        # Model info
        with st.expander("📊 Model Information"):
            st.write(f"**Features used:** {len(feature_names)}")
            st.write(f"**Linear Regression R²:** {lr_model.score.__doc__}")
            st.write(f"**Neural Network Layers:** {nn_model.hidden_layer_sizes}")
    
    # Main content based on selected mode
    if app_mode == "🏠 Home":
        st.markdown("""
        ## Welcome to the Composite Material Predictor!
        
        This application uses machine learning to predict the tensile strength of composite materials
        based on their composition and processing parameters.
        
        ### How to Use:
        1. **Manual Input**: Adjust sliders for each parameter in the Manual Input section
        2. **File Upload**: Upload a CSV file with multiple samples
        3. **Get Predictions**: Choose your model and get instant predictions
        4. **Download Results**: Export your predictions for further analysis
        
        ### Features:
        - Two ML models: Linear Regression and Neural Network
        - Interactive parameter adjustment
        - Batch prediction for multiple samples
        - Visualization of results
        - Export capabilities
        """)
        
        # Show sample data
        with st.expander("📋 Sample Data Format"):
            sample_data = pd.DataFrame({
                'Feature': feature_names,
                'Min Value': [1000, 1800, 50, 30, 15, 100, 0, 50],
                'Max Value': [3000, 2200, 2000, 200, 30, 400, 1500, 500],
                'Typical Value': [2000, 2000, 1000, 100, 22, 300, 500, 220]
            })
            st.dataframe(sample_data, use_container_width=True)
    
    elif app_mode == "📊 Manual Input":
        st.markdown('<h2 class="sub-header">Manual Parameter Input</h2>', unsafe_allow_html=True)
        
        # Create input columns
        cols = st.columns(4)
        input_values = {}
        
        for idx, feature in enumerate(feature_names):
            with cols[idx % 4]:
                # Set appropriate ranges based on feature
                if 'ratio' in feature.lower():
                    min_val, max_val, default = 0.5, 5.0, 2.5
                elif 'Density' in feature:
                    min_val, max_val, default = 1800, 2200, 2000
                elif 'Elastic modulus' in feature:
                    min_val, max_val, default = 50, 2000, 1000
                elif 'Curing' in feature:
                    min_val, max_val, default = 30, 200, 100
                elif 'Epoxy' in feature:
                    min_val, max_val, default = 15, 30, 22
                elif 'Flash' in feature:
                    min_val, max_val, default = 100, 400, 300
                elif 'Areal' in feature:
                    min_val, max_val, default = 0, 1500, 500
                elif 'Resin' in feature:
                    min_val, max_val, default = 50, 500, 220
                else:
                    min_val, max_val, default = 0, 100, 50
                
                input_values[feature] = st.number_input(
                    label=feature,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default),
                    step=0.1 if 'ratio' in feature.lower() else 1.0,
                    help=f"Enter value for {feature}"
                )
        
        # Prediction button
        if st.button("🚀 Make Prediction", type="primary"):
            # Create input dataframe
            input_df = pd.DataFrame([input_values])
            
            # Scale input
            scaled_input = scaler.transform(input_df)
            
            # Make predictions
            predictions = {}
            if selected_model in ["Linear Regression", "Both (Compare)"]:
                predictions['Linear Regression'] = lr_model.predict(scaled_input)[0]
            
            if selected_model in ["Neural Network", "Both (Compare)"]:
                predictions['Neural Network'] = nn_model.predict(scaled_input)[0]
            
            # Display results
            st.markdown("---")
            st.markdown('<h3 class="sub-header">📈 Prediction Results</h3>', unsafe_allow_html=True)
            
            if selected_model == "Both (Compare)":
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Linear Regression",
                        value=f"{predictions['Linear Regression']:.1f} MPa",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Neural Network",
                        value=f"{predictions['Neural Network']:.1f} MPa",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show difference
                diff = abs(predictions['Linear Regression'] - predictions['Neural Network'])
                st.info(f"🔍 Model difference: {diff:.1f} MPa ({diff/predictions['Linear Regression']*100:.1f}%)")
                
                # Visualization
                fig = go.Figure(data=[
                    go.Bar(name='Linear Regression', x=['Prediction'], y=[predictions['Linear Regression']]),
                    go.Bar(name='Neural Network', x=['Prediction'], y=[predictions['Neural Network']])
                ])
                fig.update_layout(title='Model Comparison', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                model_name = list(predictions.keys())[0]
                prediction_value = list(predictions.values())[0]
                
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 **Predicted Tensile Strength**")
                st.markdown(f"## **{prediction_value:.1f} MPa**")
                st.markdown(f"*Using {model_name}*")
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif app_mode == "📁 File Upload":
        st.markdown('<h2 class="sub-header">Batch Prediction from File</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CSV file with material properties",
            type=['csv', 'xlsx'],
            help="Ensure your file has the required columns"
        )
        
        if uploaded_file is not None:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)
                
                # Check columns
                missing_cols = [col for col in feature_names if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    st.info("Please ensure your file contains all required columns.")
                else:
                    st.success(f"✅ File loaded successfully: {len(df_upload)} samples")
                    
                    # Show preview
                    with st.expander("📋 Preview Data"):
                        st.dataframe(df_upload[feature_names].head(), use_container_width=True)
                    
                    # Make predictions
                    if st.button("📊 Run Batch Predictions", type="primary"):
                        with st.spinner("Processing predictions..."):
                            # Scale data
                            scaled_data = scaler.transform(df_upload[feature_names])
                            
                            # Make predictions
                            results = df_upload.copy()
                            
                            if selected_model in ["Linear Regression", "Both (Compare)"]:
                                results['LR_Prediction_MPa'] = lr_model.predict(scaled_data)
                            
                            if selected_model in ["Neural Network", "Both (Compare)"]:
                                results['NN_Prediction_MPa'] = nn_model.predict(scaled_data)
                            
                            # Display results
                            st.markdown("---")
                            st.markdown(f"### 📈 Batch Prediction Results ({len(results)} samples)")
                            
                            # Show table
                            st.dataframe(results, use_container_width=True)
                            
                            # Download button
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="composite_predictions.csv",
                                mime="text/csv",
                                type="primary"
                            )
                            
                            # Visualizations
                            if len(results) > 1:
                                st.markdown("### 📊 Distribution of Predictions")
                                
                                if selected_model == "Both (Compare)":
                                    fig = px.histogram(
                                        results,
                                        x=['LR_Prediction_MPa', 'NN_Prediction_MPa'],
                                        barmode='overlay',
                                        title='Comparison of Model Predictions',
                                        labels={'value': 'Tensile Strength (MPa)'}
                                    )
                                else:
                                    col_name = 'LR_Prediction_MPa' if selected_model == "Linear Regression" else 'NN_Prediction_MPa'
                                    fig = px.histogram(
                                        results,
                                        x=col_name,
                                        title=f'Distribution of {selected_model} Predictions',
                                        labels={col_name: 'Tensile Strength (MPa)'}
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Template download
        st.markdown("---")
        st.markdown("### 📋 Need a template?")
        
        # Create template DataFrame
        template_df = pd.DataFrame(columns=feature_names)
        template_csv = template_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download CSV Template",
            data=template_csv,
            file_name="composite_template.csv",
            mime="text/csv"
        )
    
    elif app_mode == "📈 Model Analysis":
        st.markdown('<h2 class="sub-header">Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        if 'test_predictions' in artifacts:
            test_data = artifacts['test_predictions']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Linear Regression Performance")
                # Calculate metrics
                mae_lr = np.mean(np.abs(test_data['y_test'] - test_data['y_pred_lr']))
                rmse_lr = np.sqrt(np.mean((test_data['y_test'] - test_data['y_pred_lr'])**2))
                r2_lr = 1 - np.sum((test_data['y_test'] - test_data['y_pred_lr'])**2) / np.sum((test_data['y_test'] - np.mean(test_data['y_test']))**2)
                
                st.metric("MAE", f"{mae_lr:.1f} MPa")
                st.metric("RMSE", f"{rmse_lr:.1f} MPa")
                st.metric("R² Score", f"{r2_lr:.4f}")
            
            with col2:
                st.markdown("### Neural Network Performance")
                # Calculate metrics
                mae_nn = np.mean(np.abs(test_data['y_test'] - test_data['y_pred_nn']))
                rmse_nn = np.sqrt(np.mean((test_data['y_test'] - test_data['y_pred_nn'])**2))
                r2_nn = 1 - np.sum((test_data['y_test'] - test_data['y_pred_nn'])**2) / np.sum((test_data['y_test'] - np.mean(test_data['y_test']))**2)
                
                st.metric("MAE", f"{mae_nn:.1f} MPa")
                st.metric("RMSE", f"{rmse_nn:.1f} MPa")
                st.metric("R² Score", f"{r2_nn:.4f}")
            
            # Scatter plot comparison
            st.markdown("### 📊 Actual vs Predicted")
            
            fig = go.Figure()
            
            # Add traces for both models
            fig.add_trace(go.Scatter(
                x=test_data['y_test'],
                y=test_data['y_pred_lr'],
                mode='markers',
                name='Linear Regression',
                marker=dict(color='blue', size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=test_data['y_test'],
                y=test_data['y_pred_nn'],
                mode='markers',
                name='Neural Network',
                marker=dict(color='red', size=8)
            ))
            
            # Add perfect prediction line
            max_val = max(test_data['y_test'].max(), test_data['y_pred_lr'].max(), test_data['y_pred_nn'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='green', dash='dash')
            ))
            
            fig.update_layout(
                title='Actual vs Predicted Tensile Strength',
                xaxis_title='Actual Tensile Strength (MPa)',
                yaxis_title='Predicted Tensile Strength (MPa)',
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Test prediction data not available in artifacts.")
    
    elif app_mode == "ℹ️ About":
        st.markdown("""
        ## About This Project
        
        ### Purpose
        This application predicts the tensile strength of composite materials based on their
        composition and processing parameters using machine learning models.
        
        ### Technology Stack
        - **Frontend**: Streamlit for interactive web interface
        - **ML Models**: Scikit-learn (Linear Regression, Neural Network)
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly
        - **Deployment**: Streamlit Cloud
        
        ### Workflow
        1. **Model Training**: Run `train_model.py` in Google Colab
        2. **Model Storage**: Save models to GitHub
        3. **Web App**: Deploy `streamlit_app.py` on Streamlit Cloud
        4. **Access**: Share the Streamlit URL with users
        
        ### Data Description
        The models are trained on composite material data including:
        - Matrix-filler ratio
        - Density
        - Elastic modulus
        - Curing agent content
        - Epoxy group content
        - Flash point
        - Areal density
        - Resin consumption
        
        ### Source Code
        Find the complete source code on GitHub:
        [github.com/YOUR_USERNAME/YOUR_REPO](https://github.com/)
        
        ---
        
        *Developed for composite material research and analysis*
        """)

else:
    st.error("""
    ## Models Not Loaded
    
    Please ensure:
    1. You have trained the models using `train_model.py`
    2. The model files are uploaded to GitHub
    3. The GitHub URLs in the code are correctly set
    
    ### Quick Setup Instructions:
    
    ```python
    # In train_model.py (run in Google Colab):
    1. Upload your X_bp.csv
    2. Run the training script
    3. Download the model files
    
    # In this app:
    1. Update GitHub URLs to point to your repository
    2. Deploy on Streamlit Cloud
    ```
    """)
