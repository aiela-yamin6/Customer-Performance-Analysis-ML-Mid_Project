"""Small training helpers for supported regression and classification models."""
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def build_regressor(name, params=None):
    """Create one supported regressor using safe beginner-level defaults."""
    p = params or {}
    models = {"Linear Regression": LinearRegression(), "Multiple Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=float(p.get("alpha", 1.0))), "Lasso": Lasso(alpha=float(p.get("alpha", 1.0)), max_iter=5000), "Decision Tree": DecisionTreeRegressor(max_depth=p.get("max_depth", 5), random_state=42), "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=p.get("max_depth", 8), random_state=42, n_jobs=-1)}
    if name not in models: raise ValueError("Choose Linear Regression, Ridge, Lasso, Decision Tree, or Random Forest.")
    return models[name]

def build_classifier(name, params=None):
    """Create one supported classifier from a name and simple parameters."""
    p = params or {}
    models = {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight=p.get("class_weight")), "Decision Tree": DecisionTreeClassifier(criterion=p.get("criterion", "gini"), max_depth=p.get("max_depth", 5), class_weight=p.get("class_weight"), random_state=42), "Random Forest": RandomForestClassifier(n_estimators=int(p.get("n_estimators", 100)), max_depth=p.get("max_depth", 8), class_weight=p.get("class_weight"), random_state=42, n_jobs=-1), "AdaBoost": AdaBoostClassifier(n_estimators=int(p.get("n_estimators", 80)), learning_rate=float(p.get("learning_rate", 0.8)), random_state=42), "Gradient Boosting": GradientBoostingClassifier(n_estimators=int(p.get("n_estimators", 100)), learning_rate=float(p.get("learning_rate", 0.08)), random_state=42), "SVM": SVC(C=float(p.get("C", 1)), gamma=p.get("gamma", "scale"), kernel=p.get("kernel", "rbf"), probability=True, class_weight=p.get("class_weight"), random_state=42), "Naive Bayes": GaussianNB()}
    if name not in models: raise ValueError("Choose a listed classifier.")
    return models[name]
