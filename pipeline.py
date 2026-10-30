"""Train and save reproducible best baseline models."""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from .data_ingestion import load_data
from .data_preprocessing import prepare_data
from .train import build_regressor, build_classifier
from .config import MODEL_DIR, RANDOM_STATE, TEST_SIZE, SPEND_COLUMNS

def datasets():
    """Load clean data and make independent split-first task datasets."""
    d = prepare_data(load_data()).dropna(subset=["Response", "Total_Spend"])
    cls = d.drop(columns=SPEND_COLUMNS + ["Total_Spend", "Response"])
    reg = d.drop(columns=SPEND_COLUMNS + ["Total_Spend", "Response"])
    return d, cls, reg

def make_pipeline(X, model, scale=True, imputer="median"):
    """Build train-fitted numeric and categorical preprocessing around a model."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    strategy = "median" if imputer in ["indicator", "drop rows"] else imputer
    numeric_imputer = SimpleImputer(strategy=strategy, add_indicator=(imputer == "indicator"))
    transformers = [("num", Pipeline([("imputer", numeric_imputer), ("scale", StandardScaler() if scale else "passthrough")]), numeric)]
    if "Education" in categorical:
        education_order = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
        transformers.append(("education", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OrdinalEncoder(categories=[education_order], handle_unknown="use_encoded_value", unknown_value=-1))]), ["Education"]))
    other_categories = [column for column in categorical if column != "Education"]
    if other_categories:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), other_categories))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("model", model)])

def run_pipeline():
    """Train saved best regression and classification models on fixed splits."""
    data, cls, reg = datasets()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Xr = reg
    yr = data["Total_Spend"]
    xr_train, xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    rp = make_pipeline(xr_train, build_regressor("Random Forest"))
    rp.fit(xr_train, yr_train)
    joblib.dump(rp, MODEL_DIR / "best_regressor.joblib")
    Xc, yc = cls, data["Response"].astype(int)
    xc_train, xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=yc)
    candidates = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", "SVM", "Naive Bayes"]
    best, best_auc = None, -1
    for name in candidates:
        pipe = make_pipeline(xc_train, build_classifier(name))
        pipe.fit(xc_train, yc_train)
        score = roc_auc_score(yc_test, pipe.predict_proba(xc_test)[:, 1])
        if score > best_auc: best, best_auc = pipe, score
    joblib.dump(best, MODEL_DIR / "best_classifier.joblib")
    return {"regressor": str(MODEL_DIR / "best_regressor.joblib"), "classifier": str(MODEL_DIR / "best_classifier.joblib"), "classifier_auc": float(best_auc)}
