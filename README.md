# 📱 Mobile Price Classification using Machine Learning

This project builds a machine learning system to classify mobile phones into **four price ranges** based on their hardware specifications.  
It covers end-to-end ML workflow from **EDA ➝ preprocessing ➝ model training ➝ model comparison ➝ saving the best model**.

---

## 🚀 Project Overview
The goal is to predict the **price range** of a mobile phone (0–3) using features such as:
- RAM  
- Battery power  
- Internal memory  
- Camera quality  
- Screen resolution  
- Connectivity specs  
…and more.

This project evaluates multiple machine learning models and selects the best-performing one for deployment.

---

## 📂 Dataset
The dataset used is the **Mobile Price Classification** dataset from Kaggle.

### **Target Column**
`price_range` — A categorical variable with 4 classes:
- **0** → Low cost  
- **1** → Medium cost  
- **2** → High cost  
- **3** → Very high cost  

---

## 🧹 Data Preprocessing
✔️ Removed duplicates  
✔️ Checked missing values (none found)  
✔️ Statistical summary and dataset information  
✔️ Feature scaling using **StandardScaler**

---

## 📊 Exploratory Data Analysis (EDA)
The notebook includes:

- Histograms for all features  
- Correlation heatmap  
- Target distribution plot  
- Statistical insights from `.describe()` and `.info()`  

---

## 🤖 Models Trained
A wide range of algorithms were tested:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- SVM  
- K-Nearest Neighbors  
- Naïve Bayes  
- Gradient Boosting  
- XGBoost  
- CatBoost  
- LightGBM  
- Extra Trees Classifier  

Each model's performance metrics were recorded and compared.

---

## 🏆 Best Model
The best-performing model was selected based on **accuracy**, **evaluation metrics**, and **feature importance**.

Model and scaler were saved using `pickle`:
