# streamlit_app.py - Updated with cleaner loading and credits
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
import time

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
    .credits {
        background-color: #1E293B;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
    }
    .social-links a {
        color: #60A5FA;
        text-decoration: none;
        margin: 0 10px;
    }
    .social-links a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🏗️ Composite Material Tensile Strength Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
Predict tensile strength of composite materials using machine learning models trained on composition data.
Upload your data or use the interactive controls below.
""")

# Initialize session state for loading status
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'artifacts' not in st.session_state:
    st.session_state.artifacts = None

# Load models from GitHub with progress indicator
@st.cache_resource
def load_models_from_github():
    """Load models from GitHub repository"""
    # Show loading indicator
    with st.spinner('🔄 Loading ML models from GitHub...'):
        try:
            # GitHub raw URLs
            github_username = "Areebrizz"
            repo_name = "Material-Property-Prediction-using-ML"
            branch = "main"
            
            base_url = f"https://raw.githubusercontent.com/{github_username}/{repo_name}/{branch}/"
            
            artifacts = {}
            
            # Try to load the combined artifacts first
            try:
                artifacts_url = base_url + 'model_artifacts.pkl'
                with urllib.request.urlopen(artifacts_url) as response:
                    artifacts = joblib.load(response)
                
            except Exception as e:
                # Load individual files
                # Load scaler
                scaler_url = base_url + 'scaler.pkl'
                with urllib.request.urlopen(scaler_url) as response:
                    artifacts['scaler'] = joblib.load(response)
                
                # Load linear regression model
                lr_url = base_url + 'linear_regression_model.pkl'
                with urllib.request.urlopen(lr_url) as response:
                    artifacts['lr_model'] = joblib.load(response)
                
                # Load neural network model
                nn_url = base_url + 'neural_network_model.pkl'
                with urllib.request.urlopen(nn_url) as response:
                    artifacts['nn_model'] = joblib.load(response)
                
                # Load feature names
                features_url = base_url + 'feature_names.pkl'
                with urllib.request.urlopen(features_url) as response:
                    artifacts['feature_names'] = pickle.load(response)
            
            # Store in session state
            st.session_state.artifacts = artifacts
            st.session_state.models_loaded = True
            
            return artifacts
        
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return None

# Initialize models (load once)
if not st.session_state.models_loaded:
    artifacts = load_models_from_github()
else:
    artifacts = st.session_state.artifacts

if artifacts:
    # Extract models from artifacts
    scaler = artifacts.get('scaler')
    lr_model = artifacts.get('lr_model')
    nn_model = artifacts.get('nn_model')
    feature_names = artifacts.get('feature_names')
    
    if all([scaler, lr_model, nn_model, feature_names]):
        # Success message that disappears quickly
        success_placeholder = st.empty()
        with success_placeholder:
            success = st.success('✅ All models loaded successfully!')
            time.sleep(2)  # Show for 2 seconds
        success_placeholder.empty()  # Remove the message
        
        # Sidebar with credits at the bottom
        with st.sidebar:
            st.markdown("## 🔧 Navigation")
            app_mode = st.radio(
                "Choose Mode:",
                ["🏠 Home", "📊 Manual Input", "📁 File Upload", "📈 Model Analysis", "ℹ️ About & Credits"]
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
                st.write(f"**Neural Network Layers:** {nn_model.hidden_layer_sizes}")
            
            # Credits in sidebar (small version)
            st.markdown("---")
            st.markdown("### 👨‍💻 Developer")
            st.markdown("**Muhammad Areeb Rizwan Siddiqui**")
            st.markdown("*Mechanical Engineer & Automation Specialist*")
        
        # Main content based on selected mode
        if app_mode == "🏠 Home":
            st.markdown("""
            ## Welcome to the Composite Material Predictor!
            
            This application uses machine learning to predict the tensile strength of composite materials
            based on their composition and processing parameters.
            
            ### How to Use:
            1. **Manual Input**: Adjust values for each parameter in the Manual Input section
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
            
            # Show feature names
            with st.expander("📋 Features Used in Model"):
                for i, feature in enumerate(feature_names, 1):
                    st.write(f"{i}. {feature}")
            
            # Quick prediction section
            st.markdown("---")
            st.markdown("### 🚀 Quick Start")
            st.markdown("Go to **Manual Input** section to make your first prediction!")
        
        elif app_mode == "📊 Manual Input":
            st.markdown('<h2 class="sub-header">Manual Parameter Input</h2>', unsafe_allow_html=True)
            
            # Create input columns
            cols = st.columns(3)
            input_values = {}
            
            for idx, feature in enumerate(feature_names):
                col_idx = idx % 3
                with cols[col_idx]:
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
                    st.info(f"🔍 Model difference: {diff:.1f} MPa")
                    
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
                type=['csv'],
                help="Ensure your file has the required columns"
            )
            
            if uploaded_file is not None:
                try:
                    # Read file
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Check if required columns are present
                    if all(col in df_upload.columns for col in feature_names):
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
                    
                    else:
                        missing_cols = [col for col in feature_names if col not in df_upload.columns]
                        st.error(f"❌ Missing required columns: {missing_cols}")
                        st.info("Please ensure your file contains all required columns.")
                
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
            st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Linear Regression")
                st.write("A simple linear model that finds the best linear relationship between features and target.")
                st.write("**Advantages:** Fast, interpretable, works well with linear relationships")
                
            with col2:
                st.markdown("### Neural Network")
                st.write(f"Multi-layer perceptron with architecture: {nn_model.hidden_layer_sizes}")
                st.write("**Advantages:** Can capture complex non-linear relationships")
            
            st.markdown("---")
            st.markdown("### 📋 Features Used")
            st.write(f"Total features: {len(feature_names)}")
            for feature in feature_names:
                st.write(f"- {feature}")
        
        elif app_mode == "ℹ️ About & Credits":
            col1, col2 = st.columns([2, 1])
            
            with col1:
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
                
                ### Repository
                Find the complete source code on GitHub:
                [github.com/Areebrizz/Material-Property-Prediction-using-ML](https://github.com/Areebrizz/Material-Property-Prediction-using-ML)
                """)
            
            with col2:
                st.markdown("""
                ## 🏆 Credits
                
                **Developed by:**
                ### Muhammad Areeb Rizwan Siddiqui
                
                **Mechanical Engineer**  
                **Automation Specialist**  
                **Digital Manufacturing Enthusiast**
                """)
            
            # Full credits section
            st.markdown("---")
            st.markdown('<div class="credits">', unsafe_allow_html=True)
            st.markdown("### 👨‍💻 About the Developer")
            st.markdown("""
            **Muhammad Areeb Rizwan Siddiqui** is a passionate mechanical engineer with expertise in 
            automation, digital manufacturing, and machine learning applications in materials science.
            
            Combining traditional engineering principles with modern data science techniques to solve 
            complex material property prediction challenges.
            """)
            
            st.markdown("### 🔗 Connect with Me")
            st.markdown('<div class="social-links">', unsafe_allow_html=True)
            st.markdown("""
            📧 **Email**: engr.areebriz@gmail.com  
            🔗 **LinkedIn**: [www.linkedin.com/in/areebrizwan](https://www.linkedin.com/in/areebrizwan)  
            🌐 **Website**: [areebrizwan.com](https://areebrizwan.com)  
            💼 **GitHub**: [github.com/Areebrizz](https://github.com/Areebrizz)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            ---
            *This project combines materials engineering with machine learning to enable 
            faster, data-driven material design decisions.*
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.error("Some models failed to load. Please check your model files.")
else:
    # Only show error if models truly failed to load
    if st.session_state.models_loaded == False:
        st.error("""
        ## Models Not Loaded
        
        Please ensure:
        1. The model files are in your GitHub repository
        2. The files are in the main branch
        3. The repository is public
        
        ### Files needed in GitHub:
        - model_artifacts.pkl
        - scaler.pkl
        - linear_regression_model.pkl
        - neural_network_model.pkl
        - feature_names.pkl
        """)

# Footer with minimal credits
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Composite Material Tensile Strength Predictor | Built with Streamlit</p>
    <p>Developed by Muhammad Areeb Rizwan Siddiqui • Mechanical Engineer & Automation Specialist</p>
</div>
""", unsafe_allow_html=True)
