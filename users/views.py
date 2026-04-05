from django.shortcuts import render
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
import os
import pandas as pd
import pickle
from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

# 🔥 FULL FEATURES
features = [
    'MAGE','MEDUC','GAINED','VISITS','TOTALP','BDEAD','TERMS','LOUTCOME','WEEKS',
    'RACEMOM','HISPMOM','CIGNUM','DRINKNUM',
    'ANEMIA','CARDIAC','ACLUNG','DIABETES','HERPES',
    'HYDRAM','HEMOGLOBIN','HYPERCH','HYPERPR','ECLAMP',
    'PRETERM','RENAL','RHSEN','UTERINE',
    'FAGE','FEDUC','RACEDAD','HISPDAD'
]

# ---------------- USER REGISTER ----------------
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


# ---------------- LOGIN ----------------
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)

            if check.status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                return render(request, 'users/UserHome.html')
            else:
                messages.success(request, 'Account Not Activated')
        except:
            messages.success(request, 'Invalid Login')

    return render(request, 'UserLogin.html')


# ---------------- HOME ----------------
def UserHome(request):
    return render(request, 'users/UserHome.html')


# ---------------- DATASET ----------------
def DatasetView(request):
    path = os.path.join(settings.MEDIA_ROOT, 'baby-weights_balanced_dataset.csv')
    df = pd.read_csv(path)
    return render(request, 'users/viewdataset.html', {'data': df.to_html()})


# ---------------- TRAINING ----------------
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def training(request):
    try:
        path = os.path.join(settings.MEDIA_ROOT, 'baby-weights_balanced_dataset.csv')
        df = pd.read_csv(path)

        # CLEANING
        df = df.dropna()
        df = df.drop(columns=['SEX'], errors='ignore')

        df = df.replace({'Y': 1, 'N': 0, 'M': 1, 'F': 0})
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        # TARGET
        df['TARGET'] = (
            (df['WEEKS'] < 37) |
            (df['GAINED'] < 8) |
            (df['ANEMIA'] == 1) |
            (df['DIABETES'] == 1)
        ).astype(int)

        # FEATURES (FULL)
        features = [
            'MAGE','MEDUC','GAINED','VISITS','TOTALP','BDEAD','TERMS','LOUTCOME','WEEKS',
            'RACEMOM','HISPMOM','CIGNUM','DRINKNUM',
            'ANEMIA','CARDIAC','ACLUNG','DIABETES','HERPES',
            'HYDRAM','HEMOGLOBIN','HYPERCH','HYPERPR','ECLAMP',
            'PRETERM','RENAL','RHSEN','UTERINE',
            'FAGE','FEDUC','RACEDAD','HISPDAD'
        ]

        X = df[features]
        y = df['TARGET']

        # SPLIT
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 🔥 PIPELINE (important for SVM)
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "SVM": Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(probability=True))
            ]),
            "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, eval_metric='logloss')
        }

        results = {}

        for name, model in models.items():
            # TRAIN
            model.fit(X_train, y_train)

            # TEST PREDICTION
            y_pred = model.predict(X_test)

            # CROSS VALIDATION (REAL PERFORMANCE)
            cv_scores = cross_val_score(model, X, y, cv=5)

            results[name] = {
                "accuracy": round(accuracy_score(y_test, y_pred)*100, 2),
                "precision": round(precision_score(y_test, y_pred)*100, 2),
                "recall": round(recall_score(y_test, y_pred)*100, 2),
                "f1": round(f1_score(y_test, y_pred)*100, 2),
                "cv_score": round(cv_scores.mean()*100, 2)   # ⭐ IMPORTANT
            }

        # BEST MODEL (based on CV score - REALISTIC)
        best_model_name = max(results, key=lambda x: results[x]['cv_score'])
        best_model = models[best_model_name]

        # SAVE BEST MODEL
        pickle.dump(best_model, open('model.pkl', 'wb'))

        return render(request, 'users/training.html', {
            'results': results,
            'best_model': best_model_name
        })

    except Exception as e:
        return render(request, 'users/training.html', {'error': str(e)})


# ---------------- PREDICTION ----------------
# ---------------- PREDICTION ----------------
def prediction(request):
    if request.method == 'POST':
        try:
            model = pickle.load(open('model.pkl', 'rb'))

            # IMPORTANT VALUES (for reasons)
            GAINED = float(request.POST.get('gained', 0))
            WEEKS = float(request.POST.get('weeks', 0))
            HEMOGLOBIN = float(request.POST.get('hemoglobin', 0))

            ANEMIA = int(request.POST.get('anemia', 0))
            DIABETES = int(request.POST.get('diabetes', 0))
            CARDIAC = int(request.POST.get('cardiac', 0))
            HYPERPR = int(request.POST.get('hyperpr', 0))
            CIGNUM = int(request.POST.get('cignum', 0))
            DRINKNUM = int(request.POST.get('drinknum', 0))

            # 🔥 FULL INPUT (31 FEATURES MATCH)
            input_data = [[
                float(request.POST.get('mage', 0)),
                float(request.POST.get('meduc', 0)),
                float(request.POST.get('gained', 0)),
                float(request.POST.get('visits', 0)),
                float(request.POST.get('totalp', 0)),
                float(request.POST.get('bdead', 0)),
                float(request.POST.get('terms', 0)),
                float(request.POST.get('loutcome', 0)),
                float(request.POST.get('weeks', 0)),
                float(request.POST.get('racemom', 0)),
                float(request.POST.get('hispmom', 0)),
                float(request.POST.get('cignum', 0)),
                float(request.POST.get('drinknum', 0)),
                int(request.POST.get('anemia', 0)),
                int(request.POST.get('cardiac', 0)),
                int(request.POST.get('aclung', 0)),
                int(request.POST.get('diabetes', 0)),
                int(request.POST.get('herpes', 0)),
                int(request.POST.get('hydram', 0)),
                float(request.POST.get('hemoglobin', 0)),
                int(request.POST.get('hyperch', 0)),
                int(request.POST.get('hyperpr', 0)),
                int(request.POST.get('eclamp', 0)),
                int(request.POST.get('preterm', 0)),
                int(request.POST.get('renal', 0)),
                int(request.POST.get('rhsen', 0)),
                int(request.POST.get('uterine', 0)),
                float(request.POST.get('fage', 0)),
                float(request.POST.get('feduc', 0)),
                float(request.POST.get('racedad', 0)),
                float(request.POST.get('hispdad', 0))
            ]]

            result = model.predict(input_data)

            reasons = []
            diet = []

            # 🔴 LOW WEIGHT
            if result[0] == 1:
                output = "⚠️ Low Birth Weight Risk"
                color = "red"

                if HEMOGLOBIN < 11:
                    reasons.append("Hemoglobin level is low")
                if ANEMIA == 1:
                    reasons.append("Mother has anemia")
                if CARDIAC == 1:
                    reasons.append("Mother has cardiac problems")
                if HYPERPR == 1:
                    reasons.append("Mother has high blood pressure")
                if DIABETES == 1:
                    reasons.append("Mother has diabetes")
                if GAINED < 8:
                    reasons.append("Mother weight gain is low")
                if WEEKS < 37:
                    reasons.append("Preterm delivery risk")
                if CIGNUM == 1:
                    reasons.append("Smoking affects baby growth")
                if DRINKNUM == 1:
                    reasons.append("Alcohol affects baby health")

                # 🔥 DIET
                diet = [
                    "Iron rich foods (Spinach, Beetroot)",
                    "Milk and dairy products",
                    "Protein foods (Eggs, Dal, Nuts)",
                    "Fruits (Apple, Banana)",
                    "Folic acid supplements",
                    "Avoid smoking and alcohol",
                    "Regular doctor checkups"
                ]

            # 🟢 NORMAL
            else:
                output = "✅ Normal Weight (Healthy)"
                color = "green"

                reasons.append("All parameters are normal")
                if HEMOGLOBIN >= 11:
                    reasons.append("Hemoglobin level is normal")
                if ANEMIA == 0:
                    reasons.append("No anemia")
                if DIABETES == 0:
                    reasons.append("No diabetes")
                if HYPERPR == 0:
                    reasons.append("Blood pressure is normal")

                diet = [
                    "Balanced diet",
                    "Drink plenty of water",
                    "Regular walking",
                    "Continue healthy lifestyle",
                    "Maintain good hemoglobin"
                ]

            return render(request, 'users/predictForm1.html', {
                'output': output,
                'color': color,
                'reasons': reasons,
                'diet': diet
            })

        except Exception as e:
            return render(request, 'users/predictForm1.html', {'output': str(e)})

    return render(request, 'users/predictForm1.html')
