# Network Anomaly Intrusion Detection System 🚀🔒

## Overview
This project focuses on detecting and categorizing network intrusions using Machine Learning. Network security is crucial, and this system aims to identify anomalies effectively, ensuring the protection of sensitive data and infrastructure.

## Dataset
The dataset simulates a military network environment with a mix of normal and attack traffic. Each connection, represented as a sequence of TCP packets, is labeled as either **Normal** or **Anomalous** (specific attack types). The dataset contains **41 features**:
- **38 quantitative** features
- **3 qualitative** features

🔗 **Dataset Link:** [KDD Cup 1999 Dataset](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection/data)

## Feature Selection & Data Analysis
To enhance model performance, **Mutual Information** was used for feature selection, reducing the dataset to the most impactful variables. **Exploratory Data Analysis (EDA)** was conducted using Plotly to uncover patterns and insights within the data.

## Model Building & Optimization
Several machine learning models were tested to achieve high accuracy in intrusion detection:
- **Decision Trees**
- **Random Forests**
- **Gradient Boosting** (XGBoost, LightGBM)
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Naïve Bayes**

Hyperparameter tuning was performed using **Optuna**, and model evaluation was based on:
- **Accuracy**
- **F1-score**
- **Precision & Recall**

### Best Performing Model 🚀
A **Voting Classifier** ensemble approach delivered the best results, combining the strengths of multiple models to enhance detection accuracy.

## Interactive Streamlit UI 🎛️
To make the system accessible, a **Streamlit-based UI** was developed, allowing users to:
- Upload network data
- View real-time predictions

Deployment is in progress on **Streamlit Cloud** for public accessibility.

## Project Links
- 📂 **Kaggle Notebook**: [Explore the code & insights](https://www.kaggle.com/code/nezukokamaado/intrusion-detection)

## Getting Started 🚀
### Prerequisites
Ensure you have Python installed along with the following dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/NetworkAnamolyIntrusionDetection.git
   cd NetworkAnamolyIntrusionDetection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit UI:
   ```bash
   streamlit run app.py
   ```
---
### 🔥 If you found this project useful, consider giving it a ⭐ on GitHub!

Let's build a more secure digital world together! 🔒🚀

