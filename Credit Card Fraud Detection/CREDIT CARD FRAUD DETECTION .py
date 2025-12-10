import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data_path = "Credit Card Fraud Detection/creditcard.csv/creditcard.csv"
df = pd.read_csv(data_path)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train_scaled, y_train)

def evaluate(model, xtest, ytest, name):
    print(f"\n{name}")
    preds = model.predict(xtest)
    print(classification_report(ytest, preds))
    print(confusion_matrix(ytest, preds))

evaluate(log_reg, X_test_scaled, y_test, "LOGISTIC REGRESSION")
evaluate(rf, X_test_scaled, y_test, "RANDOM FOREST")
