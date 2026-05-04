# 🎓 Student Dropout Risk Analysis using XGBoost

This project implements an **Early Warning System (EWS)** to help universities identify students at risk of dropping out. By analyzing academic performance and socio-economic factors, the model provides actionable insights for timely academic intervention.

## 🚀 Project Overview
*   **Goal**: Predict the likelihood of student dropout to support academic retention.
*   **Dataset**: 4,424 student records with 35 initial features (Grades, Attendance, Socio-economics).
*   **Algorithm**: XGBoost Classifier (Optimized with 150 estimators).
*   **Interface**: Interactive Streamlit Dashboard for real-time risk assessment.

## 📊 Performance Metrics
The model demonstrates high stability and discriminative power, validated through rigorous testing:
*   **Test Accuracy**: 93.11%
*   **Mean CV Accuracy**: 90.96% (5-fold Cross-Validation)
*   **ROC-AUC Score**: 0.97
*   **Dropout F1-Score**: 0.91

## 🛠️ Tech Stack
*   **Language**: Python 3.14
*   **ML Framework**: XGBoost, Scikit-learn
*   **Data Handling**: Pandas, NumPy
*   **Visualization**: Matplotlib
*   **Deployment**: Streamlit
*   **Model Persistence**: Joblib (using `.joblib` for optimized numerical array storage)

## 📂 Project Structure
*   `main.py`: End-to-end training pipeline, including data cleaning, feature selection, and model serialization.
*   `app.py`: Streamlit dashboard utilizing dynamic feature mapping via DataFrame inputs.
*   `student_dropout_model.joblib`: The trained XGBoost model.
*   `feature_names.joblib`: Exported feature list to ensure consistent mapping between training and inference.
*   `Students.csv`: The primary dataset.
*   `requirements.txt`: Project dependencies.

## ⚙️ How to Run
1. **Clone the repository**:
```bash
   git clone https://github.com/your-username/student-dropout-risk-analysis.git
   cd student-dropout-risk-analysis
```

2. **Install dependencies**:
```bash
   pip install -r requirements.txt
```

3. **Train the model**:
```bash
   py main.py
```

4. **Launch the dashboard**:
```bash
   streamlit run app.py
```