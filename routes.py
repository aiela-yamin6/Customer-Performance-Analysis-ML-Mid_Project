"""API endpoints call the small source module functions."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_text
from src.data_ingestion import load_data
from src.data_preprocessing import prepare_data
from src.pipeline import datasets, make_pipeline, run_pipeline
from src.config import MODEL_DIR, RANDOM_STATE, TEST_SIZE, SPEND_COLUMNS
from src.train import build_regressor, build_classifier
from src.evaluate import classification_metrics, regression_metrics

router = APIRouter()

class ModelRequest(BaseModel):
    model: str
    params: dict = Field(default_factory=dict)

class OptionRequest(BaseModel):
    method: str = "median"
    column: str = "Income"

class ScaleRequest(BaseModel):
    method: str = "standard"
    column: str = "Income"

class SelectionRequest(BaseModel):
    n_features: int = 8

class ThresholdRequest(BaseModel):
    threshold: float = Field(default=0.5, ge=0.05, le=0.95)

class PredictRequest(BaseModel):
    customer: dict
    threshold: float = Field(default=0.5, ge=0.05, le=0.95)

def safe(call):
    """Turn expected dataset or input errors into readable API messages."""
    try: return call()
    except Exception as error: raise HTTPException(status_code=400, detail=str(error))

def split_class():
    """Return stratified train and test partitions before any fitting."""
    data, X, _ = datasets(); y = data["Response"].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def split_reg():
    """Return regression partitions with all spending inputs excluded."""
    data, _, X = datasets(); y = data["Total_Spend"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

@router.get("/dataset/summary")
def summary():
    """Show dataset dimensions, types, missing data, and response balance."""
    def work():
        raw = load_data(); d = prepare_data(raw)
        return {"raw_shape": list(raw.shape), "shape": list(d.shape), "dtypes": {k:str(v) for k,v in d.dtypes.items()}, "missing": {k:int(v) for k,v in d.isna().sum().items()}, "describe": d.describe(include="all").replace({np.nan:None}).to_dict(), "response_balance": {str(k):int(v) for k,v in d.Response.value_counts().items()}, "columns": d.columns.tolist()}
    return safe(work)

@router.get("/eda")
def eda():
    """Return histogram samples and target correlations for the EDA page."""
    def work():
        d=prepare_data(load_data()); numeric=d.select_dtypes(include="number")
        return {"histograms":{c:[float(v) for v in d[c].dropna().sample(min(400,len(d)),random_state=42)] for c in ["Income","Age","Total_Spend"]}, "corr_response":{k:float(v) for k,v in numeric.corr()["Response"].drop("Response").items()}, "corr_spend":{k:float(v) for k,v in numeric.corr()["Total_Spend"].drop("Total_Spend").items()}, "balance": {str(k):int(v) for k,v in d.Response.value_counts().items()}}
    return safe(work)

@router.post("/preprocess/missing")
def missing(request: OptionRequest):
    """Compare a selected missing-value strategy on Income and show counts."""
    def work():
        d=prepare_data(load_data()); col=request.column
        if col not in d: raise ValueError("Unknown column.")
        s=d[col]; counts={"missing":int(s.isna().sum()),"percent":float(s.isna().mean()*100)}
        if request.method=="drop rows": after=s.dropna()
        elif request.method in ["mean","median","mode"]:
            strategy="most_frequent" if request.method=="mode" else request.method
            after=pd.Series(SimpleImputer(strategy=strategy).fit_transform(s.to_frame()).ravel())
        elif request.method=="indicator": after=pd.Series(SimpleImputer(strategy="median",add_indicator=True).fit_transform(s.to_frame())[:,0])
        else: raise ValueError("Choose drop rows, mean, median, mode, or indicator.")
        all_missing={name:{"count":int(values.isna().sum()),"percent":float(values.isna().mean()*100)} for name,values in d.items()}
        return {"column":col,"before":counts,"missing_by_column":all_missing,"after_missing":int(after.isna().sum()),"method":request.method,"effect_auc": logistic_auc(d, request.method)}
    return safe(work)

def logistic_auc(d, method="median", scale=True):
    """Fit logistic regression with training-only preprocessing for ROC-AUC."""
    if method == "drop rows": d=d.dropna(subset=["Income"])
    if method == "mode": method="most_frequent"
    X=d.drop(columns=SPEND_COLUMNS+["Total_Spend","Response"]); y=d.Response.astype(int)
    a,b,c,e=train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)
    return float(__import__('sklearn.metrics',fromlist=['roc_auc_score']).roc_auc_score(e,make_pipeline(a,LogisticRegression(max_iter=3000),scale=scale,imputer=method).fit(a,c).predict_proba(b)[:,1]))

@router.post("/preprocess/scaling")
def scaling(request: ScaleRequest):
    """Show summary statistics for a chosen scaling method and column."""
    def work():
        d=prepare_data(load_data()); col=request.column; x=d[[col]].dropna().astype(float)
        methods={"minmax":MinMaxScaler(),"standard":StandardScaler(),"robust":RobustScaler()}
        if request.method not in methods: raise ValueError("Choose minmax, standard, or robust.")
        z=methods[request.method].fit_transform(x)
        stats=lambda v:{"mean":float(np.mean(v)),"std":float(np.std(v)),"min":float(np.min(v)),"max":float(np.max(v))}
        return {"before":stats(x.values),"after":stats(z),"scaling_models":{"need scaling":["SVM","Logistic Regression","Ridge","Lasso","PCA"],"usually do not need scaling":["Trees","Forests","Boosting","Naive Bayes"]}}
    return safe(work)

@router.post("/preprocess/encoding")
def encoding():
    """Return encoded category columns using the course's ordinal order."""
    return {"education_order":["Basic","2n Cycle","Graduation","Master","PhD"],"marital_values":sorted(prepare_data(load_data()).Marital_Status.dropna().unique().tolist()),"one_hot_columns":["Marital_Status_"+str(x) for x in sorted(prepare_data(load_data()).Marital_Status.dropna().unique())]}

@router.post("/preprocess/selection")
def selection(request: SelectionRequest):
    """Compare RFE, forest importance, and permutation rankings."""
    def work():
        a,b,c,e=split_class(); model=LogisticRegression(max_iter=2000)
        pipe=make_pipeline(a,model).fit(a,c)
        # Use transformed numeric feature names for interpretable comparison.
        names=pipe.named_steps["prep"].get_feature_names_out().tolist(); xx=pipe.named_steps["prep"].transform(a)
        rfe=RFE(LogisticRegression(max_iter=2000),n_features_to_select=min(request.n_features,len(names))).fit(xx,c)
        forest=build_classifier("Random Forest").fit(xx,c); perm=permutation_importance(forest,pipe.named_steps["prep"].transform(b),e,n_repeats=3,random_state=42,scoring="roc_auc")
        return {"rfe":sorted(zip(names,rfe.ranking_.tolist()),key=lambda x:x[1])[:20],"forest":sorted(zip(names,forest.feature_importances_.tolist()),key=lambda x:x[1],reverse=True)[:20],"permutation":sorted(zip(names,perm.importances_mean.tolist()),key=lambda x:x[1],reverse=True)[:20]}
    return safe(work)

@router.post("/preprocess/pca")
def pca():
    """Return PCA explained variance and 90 percent component count."""
    def work():
        a,b,c,e=split_class(); numeric=a.select_dtypes(include="number").columns; imp=SimpleImputer(strategy="median"); scaler=StandardScaler(); train=scaler.fit_transform(imp.fit_transform(a[numeric])); test=scaler.transform(imp.transform(b[numeric])); model=PCA().fit(train); count=int(np.searchsorted(np.cumsum(model.explained_variance_ratio_),.9)+1)
        baseline=LogisticRegression(max_iter=2000).fit(train,c); reduced=PCA(n_components=count).fit(train); after=LogisticRegression(max_iter=2000).fit(reduced.transform(train),c)
        return {"variance":[float(x) for x in model.explained_variance_ratio_],"components_90":count,"accuracy_before":float(accuracy_score(e,baseline.predict(test))),"accuracy_after":float(accuracy_score(e,after.predict(reduced.transform(test))))}
    return safe(work)

@router.post("/preprocess/pipeline")
def preprocessing_pipeline():
    """Compare baseline and complete train-fitted logistic pipelines."""
    def work():
        a,b,c,e=split_class(); full=make_pipeline(a,LogisticRegression(max_iter=2000)).fit(a,c); simple=make_pipeline(a,LogisticRegression(max_iter=2000),scale=False).fit(a,c)
        from sklearn.metrics import roc_auc_score
        baseline_probability=simple.predict_proba(b)[:,1]
        full_probability=full.predict_proba(b)[:,1]
        baseline_fpr,baseline_tpr,_=roc_curve(e,baseline_probability)
        full_fpr,full_tpr,_=roc_curve(e,full_probability)
        return {"baseline_auc":float(roc_auc_score(e,baseline_probability)),"full_pipeline_auc":float(roc_auc_score(e,full_probability)),"baseline_roc":{"fpr":[float(v) for v in baseline_fpr],"tpr":[float(v) for v in baseline_tpr]},"full_roc":{"fpr":[float(v) for v in full_fpr],"tpr":[float(v) for v in full_tpr]}}
    return safe(work)

@router.post("/regression/train")
def regression_train(req: ModelRequest):
    """Train the selected regressor on the fixed regression split."""
    def work():
        a,b,c,e=split_reg()
        if req.model in ["Simple Linear Regression", "Polynomial Regression"]:
            from sklearn.pipeline import Pipeline
            degree=1 if req.model=="Simple Linear Regression" else int(req.params.get("degree",3))
            if degree<1 or degree>10: raise ValueError("Polynomial degree must be between 1 and 10.")
            a=a[["Income"]]; b=b[["Income"]]
            pipe=Pipeline([("impute",SimpleImputer(strategy="median")),("poly",PolynomialFeatures(degree=degree,include_bias=False)),("model",LinearRegression())]).fit(a,c)
            coefficients=[{"feature":f"Income^{i+1}","value":float(v)} for i,v in enumerate(pipe.named_steps["model"].coef_)]
            ordered=np.linspace(float(a.Income.min()),float(a.Income.max()),100).reshape(-1,1)
            curve={"income":[float(v) for v in ordered.ravel()],"spend":[float(v) for v in pipe.predict(ordered)]}
        else:
            pipe=make_pipeline(a,build_regressor(req.model,req.params)).fit(a,c)
            coefficients=coef_data(pipe,a.columns); curve={}
        return {"model":req.model,"train":regression_metrics(c,pipe.predict(a)),"test":regression_metrics(e,pipe.predict(b)),"coefficients":coefficients,"curve":curve,"feature_importance":importance_data(pipe,a.columns)}
    return safe(work)

def coef_data(pipe, columns):
    """Return model coefficient pairs when the selected model has them."""
    model=pipe.named_steps["model"]
    if not hasattr(model,"coef_"): return []
    prep=pipe.named_steps.get("prep")
    names=prep.get_feature_names_out() if prep is not None else columns
    return [{"feature":str(k),"value":float(v)} for k,v in zip(names,model.coef_)]

def importance_data(pipe, columns):
    """Return feature importance pairs when the selected model provides them."""
    model=pipe.named_steps["model"]
    if not hasattr(model,"feature_importances_"): return []
    prep=pipe.named_steps.get("prep")
    names=prep.get_feature_names_out() if prep is not None else columns
    return [{"feature":str(k),"value":float(v)} for k,v in zip(names,model.feature_importances_)]

@router.get("/regression/compare")
def regression_compare():
    """Evaluate the supported regressors and save a decision tree chart payload."""
    def work():
        a,b,c,e=split_reg(); rows=[]; best=None
        for name in ["Simple Linear Regression", "Linear Regression","Ridge","Lasso","Polynomial Regression","Decision Tree","Random Forest"]:
            if name in ["Simple Linear Regression", "Polynomial Regression"]:
                from sklearn.pipeline import Pipeline
                degree=1 if name=="Simple Linear Regression" else 3
                aa,bb=a[["Income"]],b[["Income"]]
                pipe=Pipeline([("impute",SimpleImputer(strategy="median")),("poly",PolynomialFeatures(degree=degree,include_bias=False)),("model",LinearRegression())]).fit(aa,c)
            else: aa,bb=a,b; pipe=make_pipeline(aa,build_regressor(name)).fit(aa,c)
            tr=regression_metrics(c,pipe.predict(aa)); te=regression_metrics(e,pipe.predict(bb)); rows.append({"model":name,"train":tr,"test":te})
            if best is None or te["r2"]>best["test"]["r2"]: best={"model":name,"train":tr,"test":te}
        return {"models":rows,"best":best}
    return safe(work)

@router.post("/classification/train")
def classification_train(req: ModelRequest):
    """Train a classifier, then return metrics, curve points, and tree rules."""
    def work():
        a,b,c,e=split_class(); model=build_classifier(req.model,req.params); pipe=make_pipeline(a,model,scale=req.model in ["SVM","Logistic Regression"]).fit(a,c); prob=pipe.predict_proba(b)[:,1]; pred=(prob>=float(req.params.get("threshold",.5))).astype(int); metrics=classification_metrics(e,pred,prob); fpr,tpr,_=roc_curve(e,prob)
        rules=export_text(pipe.named_steps["model"],max_depth=4) if req.model=="Decision Tree" else ""
        return {"model":req.model,"metrics":metrics,"roc":{"fpr":[float(x) for x in fpr],"tpr":[float(x) for x in tpr]},"rules":rules,"importance":importance_data(pipe,a.columns)}
    return safe(work)

@router.get("/classification/compare")
def classification_compare():
    """Compare supported classifiers and save the highest-AUC model."""
    def work():
        a,b,c,e=split_class(); rows=[]; curves=[]; best=None
        for name in ["Logistic Regression","Decision Tree","Random Forest","AdaBoost","Gradient Boosting","SVM","Naive Bayes"]:
            pipe=make_pipeline(a,build_classifier(name),scale=name in ["SVM","Logistic Regression"]).fit(a,c); prob=pipe.predict_proba(b)[:,1]; pred=(prob>=.5).astype(int); metrics=classification_metrics(e,pred,prob); rows.append({"model":name,**metrics}); fpr,tpr,_=roc_curve(e,prob); curves.append({"model":name,"fpr":[float(x) for x in fpr],"tpr":[float(x) for x in tpr]})
            if best is None or metrics["roc_auc"]>best["roc_auc"]: best={"model":name,"roc_auc":metrics["roc_auc"]}
        return {"models":rows,"curves":curves,"best":best}
    return safe(work)

@router.post("/classification/threshold")
def threshold(req: ThresholdRequest):
    """Apply a requested decision threshold to held-out logistic scores."""
    def work():
        a,b,c,e=split_class(); pipe=make_pipeline(a,build_classifier("Logistic Regression")).fit(a,c); prob=pipe.predict_proba(b)[:,1]; pred=(prob>=req.threshold).astype(int); return classification_metrics(e,pred,prob)
    return safe(work)

@router.post("/imbalance/run")
def imbalance(req: ThresholdRequest):
    """Compare class weighting and train-only SMOTE against baseline."""
    def work():
        from imblearn.over_sampling import SMOTE
        from sklearn.dummy import DummyClassifier
        a,b,c,e=split_class(); rows=[]
        for name, model, smote in [("Always no",DummyClassifier(strategy="most_frequent"),False),("Baseline",build_classifier("Logistic Regression"),False),("Balanced weight",build_classifier("Logistic Regression",{"class_weight":"balanced"}),False),("SMOTE train only",build_classifier("Logistic Regression"),True)]:
            if smote: aa,cc=SMOTE(random_state=42,k_neighbors=5).fit_resample(pd.get_dummies(a).fillna(0),c); temp=LogisticRegression(max_iter=2000).fit(aa,cc); pp=temp.predict_proba(pd.get_dummies(b).reindex(columns=aa.columns,fill_value=0).fillna(0))[:,1]
            else:
                pipe=make_pipeline(a,model).fit(a,c); pp=pipe.predict_proba(b)[:,1]
            rows.append({"model":name,**classification_metrics(e,(pp>=req.threshold if name=="Lower threshold" else pp>=.5).astype(int),pp)})
        pipe=make_pipeline(a,build_classifier("Logistic Regression")).fit(a,c); pp=pipe.predict_proba(b)[:,1]; rows.append({"model":"User threshold",**classification_metrics(e,(pp>=req.threshold).astype(int),pp)})
        return {"models":rows,"note":"Use class weights when you want a simple cost-sensitive model; use SMOTE when minority examples are too scarce for the learner; lower a threshold when missed positives cost more and review capacity allows extra alerts."}
    return safe(work)

@router.post("/predict")
def predict(req: PredictRequest):
    """Use saved best models to estimate spending and campaign response."""
    def work():
        rp=joblib.load(MODEL_DIR/"best_regressor.joblib"); cp=joblib.load(MODEL_DIR/"best_classifier.joblib"); customer=req.customer.copy()
        # The lab models use the engineered purchase total as a feature.
        customer["Total_Purchases"] = sum(float(customer.get(column, 0)) for column in ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"])
        spend=float(rp.predict(pd.DataFrame([customer]))[0]); chance=float(cp.predict_proba(pd.DataFrame([customer]))[0,1]); yes=chance>=req.threshold
        return {"total_spend":spend,"response_probability":chance,"decision":"yes" if yes else "no","business_sentence":f"{chance:.0%} chance - {'include' if yes else 'do not prioritize'} this customer in the next campaign."}
    return safe(work)

@router.get("/options")
def options():
    """List supported model names and UI parameter options."""
    return {"regressors":["Simple Linear Regression","Multiple Linear Regression","Linear Regression","Ridge","Lasso","Polynomial Regression","Decision Tree","Random Forest"],"classifiers":["Logistic Regression","Decision Tree","Random Forest","AdaBoost","Gradient Boosting","SVM","Naive Bayes"],"scalers":["minmax","standard","robust"],"imputers":["drop rows","mean","median","mode","indicator"]}

@router.on_event("startup")
def startup():
    """Train both saved models automatically on the first API start."""
    if not (MODEL_DIR/"best_regressor.joblib").exists() or not (MODEL_DIR/"best_classifier.joblib").exists(): safe(run_pipeline)
