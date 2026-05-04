import pandas as pd
import xgboost as xgb 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('Students.csv')
# display(df)
df.columns = df.columns.str.strip()

df = df.drop(columns=['Nacionality', 'International', 'Application order'])
df = df[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Dropout': 1, 'Graduate': 0})

print(f"New shape: {df.shape}")
# print(f"Remaining Columns: {df.columns.tolist()}")
# print(f"Unique Categories: {df['Target'].unique()}")

X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')

cv_score = cross_val_score(model, X, y, cv=5)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]


fig, ax = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Graduate', 'Dropout'])
disp.plot(ax=ax[0], cmap='Blues')
ax[0].set_title(f'Confusion Matrix (Acc: {accuracy_score(y_test, y_pred):.2%})')

fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)
ax[1].plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {auc_score:.2f})')
ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('ROC Curve')
ax[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

joblib.dump(model, 'student_dropout_model.joblib')
joblib.dump(X.columns.tolist(), 'feature_names.joblib')

print(f"\n--- Analysis Summary ---\n")
print(f"Mean CV Accuracy: {cv_score.mean():.2%}")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"ROC-AUC Score: {auc_score:.2f}")
print(f"\nDetailed Report:\n{classification_report(y_test, y_pred, target_names=['Graduate', 'Dropout'])}")