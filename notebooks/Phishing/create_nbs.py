import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# 1. Preprocessing Notebook
nb = new_notebook()

nb.cells.extend([
    new_markdown_cell("# 📧 SecureMind AI — Phishing Dataset Preprocessing\n\nThis notebook loads the raw phishing dataset from `data/Phishing/phishing_email.csv`, explores the basic structure, handles any missing values, and exports the clean subset to `data_output/phishing_cleaned.csv` ready for model training."),
    new_markdown_cell("## 1. Import Libraries"),
    new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print('✅ Libraries imported successfully')"""),
    new_markdown_cell("## 2. Load Raw Dataset"),
    new_code_cell("""DATA_PATH = os.path.join('../../data/Phishing/phishing_email.csv')
data = pd.read_csv(DATA_PATH)
print(f'✅ Dataset loaded: {data.shape}')
data.head()"""),
    new_markdown_cell("## 3. Exploratory Data Analysis & Cleaning"),
    new_code_cell("""data.info()"""),
    new_code_cell("""print(data.isnull().sum())

# Drop missing values if any
data = data.dropna()
print(f'\\nShape after dropping NaNs: {data.shape}')"""),
    new_code_cell("""# Rename column for consistency if desired
data.rename(columns={'text_combined': 'text'}, inplace=True)
data.head()"""),
    new_code_cell("""print(f'\\n📊 Class Distribution:')
class_dist = data['label'].value_counts()
for idx, count in class_dist.items():
    pct = count / len(data) * 100
    label_name = 'Phishing' if idx == 1 else 'Legitimate'
    print(f'  {idx} → {label_name:15s}  {count:>10,}  ({pct:.2f}%)')

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=data, palette='Set2')
plt.title('Phishing Dataset — Class Distribution')
plt.xticks([0, 1], ['Legitimate (0)', 'Phishing (1)'])
plt.show()"""),
    new_code_cell("""# Optionally view text length distributions
data['text_length'] = data['text'].apply(lambda x: len(str(x)))

plt.figure(figsize=(10, 5))
sns.histplot(data=data, x='text_length', hue='label', bins=50, kde=True)
plt.title('Text Length Distribution by Label')
plt.xlim(0, 20000)
plt.show()"""),
    new_markdown_cell("## 4. Export Cleaned Dataset"),
    new_code_cell("""export_path = os.path.join('../../data_output', 'phishing_cleaned.csv')
data[['text', 'label']].to_csv(export_path, index=False)
print(f'✅ Cleaned dataset saved to {export_path}')""")
])

with open('/home/achrafness/All/mygit/SecureMind-AI/notebooks/Phishing/data-preprocessing.ipynb', 'w') as f:
    nbformat.write(nb, f)


# 2. Model Training Notebook
nb2 = new_notebook()

nb2.cells.extend([
    new_markdown_cell("# 🛡️ SecureMind AI — Phishing Email Model Training Pipeline\n\nThis notebook uses the cleaned phishing dataset to train models mimicking the pipeline used in the CICIDS2017 workflow:\n1. Load cleaned data\n2. Train/Validation/Test split (stratified)\n3. TF-IDF Vectorization (instead of Standard Scaling)\n4. Train Decision Tree and Random Forest\n5. Evaluate on Validation and Test sets"),
    new_markdown_cell("## 1. Import Libraries"),
    new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings('ignore')
print('✅ Libraries imported successfully')"""),
    new_markdown_cell("## 2. Load Cleaned Dataset"),
    new_code_cell("""DATA_PATH = os.path.join('../../data_output/phishing_cleaned.csv')
data = pd.read_csv(DATA_PATH)
# Ensure all text is string and drop any unexpected NaNs
data = data.dropna()
data['text'] = data['text'].astype(str)

print(f'✅ Dataset loaded: {data.shape}')
data.head()"""),
    new_code_cell("""TARGET_COL = 'label'
LABEL_NAMES = {0: 'Legitimate', 1: 'Phishing'}"""),
    new_markdown_cell("## 3. Train / Validation / Test Split (70/15/15)"),
    new_code_cell("""X = data['text']
y = data[TARGET_COL]

# Stage 1: split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Stage 2: split temp into validation (50% of 30% = 15%) and test (50% of 30% = 15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

del X_temp, y_temp  # free memory

print(f'✅ Train set:      {X_train.shape[0]:>10,} samples  ({X_train.shape[0]/len(data)*100:.1f}%)')
print(f'✅ Validation set: {X_val.shape[0]:>10,} samples  ({X_val.shape[0]/len(data)*100:.1f}%)')
print(f'✅ Test set:       {X_test.shape[0]:>10,} samples  ({X_test.shape[0]/len(data)*100:.1f}%)')"""),
    new_markdown_cell("## 4. Text Vectorization (TF-IDF)"),
    new_code_cell("""vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit on train ONLY, transform all sets to prevent data leakage
start_time = time.time()
X_train_vec = vectorizer.fit_transform(X_train)
print(f'✅ Fit and transformed training data in {time.time() - start_time:.2f}s')

X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

print(f'✅ Vectorization complete. Vocabulary size: {len(vectorizer.get_feature_names_out())}')"""),
    new_markdown_cell("## 5. Model Training & Validation"),
    new_code_cell("""def train_and_evaluate(name, model, X_tr, y_tr, X_v, y_v, label_names):
    \"\"\"Train a model, evaluate on validation set, return results dict.\"\"\"
    print(f'\\n{\"=\"*60}')
    print(f'  Training: {name}')
    print(f'{\"=\"*60}')

    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start
    print(f'  ⏱️  Training time: {train_time:.2f}s')

    # Predict on validation
    start = time.time()
    y_pred = model.predict(X_v)
    pred_time = time.time() - start

    acc = accuracy_score(y_v, y_pred)
    print(f'  ✅ Validation Accuracy: {acc:.4f}  ({acc*100:.2f}%)')
    print(f'  ⏱️  Prediction time:  {pred_time:.2f}s')

    # Classification report
    target_names = [label_names[i] for i in sorted(label_names.keys())]
    report = classification_report(
        y_v, y_pred,
        target_names=target_names,
        zero_division=0
    )
    print(f'\\n{report}')

    return {
        'name': name,
        'model': model,
        'accuracy': acc,
        'train_time': train_time,
        'pred_time': pred_time,
        'y_pred': y_pred,
    }

print('✅ Helper function defined')"""),
    new_code_cell("""dt_model = DecisionTreeClassifier(random_state=42, max_depth=20)

dt_results = train_and_evaluate(
    'Decision Tree', dt_model,
    X_train_vec, y_train,
    X_val_vec, y_val,
    LABEL_NAMES
)"""),
    new_code_cell("""rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)

rf_results = train_and_evaluate(
    'Random Forest', rf_model,
    X_train_vec, y_train,
    X_val_vec, y_val,
    LABEL_NAMES
)"""),
    new_markdown_cell("## 6. Test Set Final Evaluation"),
    new_code_cell("""best_model = rf_model if rf_results['accuracy'] > dt_results['accuracy'] else dt_model
best_model_name = rf_results['name'] if rf_results['accuracy'] > dt_results['accuracy'] else dt_results['name']

print(f'🏅 Evaluating Best Model ({best_model_name}) on Held-out TEST Set:')
start = time.time()
y_test_pred = best_model.predict(X_test_vec)
print(f'  ⏱️  Test Prediction time: {time.time() - start:.2f}s')

test_acc = accuracy_score(y_test, y_test_pred)
print(f'  🎯  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')

target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
print(classification_report(y_test, y_test_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax, colorbar=False)
plt.title(f'{best_model_name} - Test Set Confusion Matrix')
plt.show()"""),
    new_markdown_cell("## 7. Export Artifacts"),
    new_code_cell("""os.makedirs('../../models/Phishing', exist_ok=True)

model_path = '../../models/Phishing/best_phishing_model.pkl'
vectorizer_path = '../../models/Phishing/tfidf_vectorizer.pkl'
label_map_path = '../../models/Phishing/label_map.pkl'

joblib.dump(best_model, model_path)
joblib.dump(vectorizer, vectorizer_path)
joblib.dump(LABEL_NAMES, label_map_path)

print(f'✅ Best model ({best_model_name}) saved to {model_path}')
print(f'✅ TF-IDF vectorizer saved to {vectorizer_path}')
print(f'✅ Label map saved to {label_map_path}')

# Save split datasets
os.makedirs('../../data_output/Phishing', exist_ok=True)
pd.DataFrame({'text': X_train, 'label': y_train}).to_csv('../../data_output/Phishing/train.csv', index=False)
pd.DataFrame({'text': X_val, 'label': y_val}).to_csv('../../data_output/Phishing/val.csv', index=False)
pd.DataFrame({'text': X_test, 'label': y_test}).to_csv('../../data_output/Phishing/test.csv', index=False)
print('✅ Split datasets saved to ../../data_output/Phishing/')""")
])

with open('/home/achrafness/All/mygit/SecureMind-AI/notebooks/Phishing/model-training.ipynb', 'w') as f:
    nbformat.write(nb2, f)
