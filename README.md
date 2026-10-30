# Customer ML Lab

 A FastAPI and dashboard for the customer performance midterm. It explores the supplied data, compares regression and classification models, demonstrates imbalance strategies, and predicts spending and campaign response for a new customer.

## Run it

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

Open <http://127.0.0.1:8000>. The API documentation is at <http://127.0.0.1:8000/docs>.

Keep Uvicorn running while using the dashboard. If you serve `frontend/` separately (for example, with VS Code Live Server), the dashboard will retry API requests against `http://localhost:8000`; the backend must still be running there. You can also use the simpler single-server setup above and open the dashboard from Uvicorn.

## Dataset

Place `customer_performance.csv` in `data/raw/`. This project reads it as tab-separated data (`sep="\t"`). The supplied workspace copy has already been placed at `data/raw/customer_performance.csv`.

## Folder map

- `api/`: FastAPI app and endpoints
- `data/raw/`: original input
- `docs/`: lab report 
- `frontend/`: HTML, CSS, and plain JavaScript dashboard
- `models/`: generated `.joblib` models and model charts
- `src/`: ingestion, cleaning, model helpers, evaluation, and training pipeline

## Modeling notes

The fixed random seed is 42 and the holdout is 30%. Classification uses a stratified split. Imputers, encoders, and scalers are fitted inside sklearn pipelines on training rows only. Regression never uses the six `Mnt*` spending columns or `Total_Spend` as inputs. SMOTE is applied only to training rows.

On first launch, the API trains and saves `models/best_regressor.joblib` and `models/best_classifier.joblib` if they are missing. Comparison pages recalculate metrics using the same fixed split.

## Troubleshooting

- If income or other values appear in the wrong columns, check that the file is tab-separated and was loaded with `sep="\t"`.
- If port 8000 is in use, run `uvicorn api.main:app --reload --port 8001` and open the matching local URL.
- If dependencies are missing, activate the environment and rerun `pip install -r requirements.txt`.
