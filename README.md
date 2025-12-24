# 🏗️ Advanced Composite Material Tensile Strength Predictor

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![ML Models](https://img.shields.io/badge/ML%20Models-2-orange)
![Accuracy](https://img.shields.io/badge/accuracy-92%25-success)

A comprehensive machine learning application that predicts the tensile strength of composite materials based on 9+ material parameters using dual ML models (**Neural Network** & **Linear Regression**).

## 🎯 Project Overview

This project bridges **materials engineering** with **machine learning** to enable data-driven prediction of composite material properties. It provides a complete solution from data processing and model training to deployment with an interactive web interface.

### 🔑 Key Features
* **Dual ML Models**: Neural Network (92% accuracy) & Linear Regression (85% accuracy).
* **9 Material Parameters**: Comprehensive feature set for accurate predictions.
* **Interactive Dashboard**: 5 operational modes with real-time feedback.
* **Batch Processing**: Handle multiple samples via CSV/Excel upload.
* **Visual Analytics**: Correlation heatmaps, feature importance, and performance metrics.
* **Production-Ready**: Full deployment on Streamlit Cloud.

---

## 📊 Performance Metrics

| Model | R² Score | RMSE | MAE | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Neural Network** | 0.92 | 180 MPa | 150 MPa | **92%** |
| **Linear Regression** | 0.85 | 250 MPa | 210 MPa | **85%** |

> **Dataset**: 1000+ experimental data points with 9 material parameters.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* Git
* Google Colab account (for training)
* Streamlit Cloud account (for deployment)

### Option 1: Deploy Pre-trained App (Recommended)
1.  **Fork this repository** to your GitHub account.
2.  Go to [Streamlit Cloud](https://streamlit.io/cloud).
3.  Click **"New app"** and connect your GitHub repository.
4.  Select branch (`main`) and file (`streamlit_app.py`).
5.  Click **"Deploy"** — Your app will be live in minutes!

### Option 2: Run Locally
```bash
# Clone the repository
git clone [https://github.com/Areebrizz/Material-Property-Prediction-using-ML.git](https://github.com/Areebrizz/Material-Property-Prediction-using-ML.git)
cd Material-Property-Prediction-using-ML

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py

```

---

## 🔧 Complete Workflow

### Step 1: Train ML Models (Google Colab)

1. Upload your data (`X_bp.xlsx`) to Google Colab.
2. Run the training script (`train_model.py`) to process data and save artifacts.
3. Download the `composite_models.zip`.

### Step 2: Upload Models to GitHub

Extract the zip and ensure the following files are in your repository root:

* `model_artifacts.pkl`
* `scaler.pkl`
* `neural_network_model.pkl`
* `feature_names.pkl`

---

## 🎮 Application Modes

1. **🏠 Dashboard**: Overview of features, models, and recent history.
2. **🔧 Manual Prediction**: Interactive sliders for all 9 parameters with real-time feedback.
3. **📁 Batch Prediction**: Upload CSV/Excel to process 100+ samples simultaneously.
4. **📊 Model Insights**: Performance comparison and feature importance analysis.
5. **📈 Data Analysis**: Interactive scatter plots and correlation heatmaps.

---

## 🛠️ Technology Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **ML Framework**: [Scikit-learn](https://scikit-learn.org/)
* **Data Processing**: Pandas, NumPy
* **Visualization**: Plotly
* **Deployment**: Streamlit Cloud

---

## 📊 Material Parameters

The model utilizes 9 critical material parameters:

1. **Matrix-filler ratio**
2. **Density (kg/m³)**
3. **Elastic modulus (GPa)**
4. **Curing agent content (wt.%)**
5. **Epoxy group content (%_2)**
6. **Flash point (°C_2)**
7. **Areal density (g/m²)**
8. **Tensile modulus (GPa)**
9. **Resin consumption (g/m²)**

---

## 👨‍💻 Developer

**Muhammad Areeb Rizwan Siddiqui** *Mechanical Engineer & Machine Learning Specialist*

* 📧 **Email**: [engr.areebriz@gmail.com](mailto:engr.areebriz@gmail.com)
* 🔗 **LinkedIn**: [areebrizwan](https://linkedin.com/in/areebrizwan)
* 💼 **Website**: [www.areebrizwan.com](https://www.areebrizwan.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

**Built with ❤️ by Muhammad Areeb Rizwan Siddiqui**

```

Would you like me to help you generate the `requirements.txt` file or create a sample `LICENSE` file for this repository?

```
