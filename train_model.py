import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import pickle

# Load dataset
data = pd.read_csv('media/baby-weights_balanced_dataset.csv')

# Clean
data = data.dropna()
data = data.drop(columns=['SEX'], errors='ignore')

# Convert categorical
data = data.replace({'Y': 1, 'N': 0, 'M': 1, 'F': 0})
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# Target
data['TARGET'] = (
    (data['WEEKS'] < 37) |
    (data['GAINED'] < 8) |
    (data['ANEMIA'] == 1) |
    (data['DIABETES'] == 1)
).astype(int)

# Features (FULL)
features = [
    'MAGE','MEDUC','GAINED','VISITS','TOTALP','BDEAD','TERMS','LOUTCOME','WEEKS',
    'RACEMOM','HISPMOM','CIGNUM','DRINKNUM',
    'ANEMIA','CARDIAC','ACLUNG','DIABETES','HERPES',
    'HYDRAM','HEMOGLOBIN','HYPERCH','HYPERPR','ECLAMP',
    'PRETERM','RENAL','RHSEN','UTERINE',
    'FAGE','FEDUC','RACEDAD','HISPDAD'
]

X = data[features]
y = data['TARGET']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

# Best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]

# Save
pickle.dump(best_model, open('model.pkl', 'wb'))

print("Results:", results)
print("Best Model:", best_model_name)
print("✅ Model Saved Successfully")
