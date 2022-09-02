import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

__dir = os.path.dirname(os.path.realpath(__file__))

__df = pd.read_csv(f'{__dir}/diabetes.csv', encoding='utf-8', header=None)
__df.columns = [
    "TimesPregnant",
    "PlasmaGlucoseConcentration",
    "BloodPressure",
    "SkinfoldThickness",
    "Insulin",
    "BodyMassIndex",
    "DiabetesPedigreeFunction",
    "Age",
    "HasDiabetes"
]

config = {
    "dataset": {
        "samples": __df.drop(["HasDiabetes"], axis=1),
        "labels": __df.HasDiabetes,
    },
    "validation": {
        "method": "loo"
    },
    "models": [
        LogisticRegression(C=1, max_iter=500),
        KNeighborsClassifier(3),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    ]
}
