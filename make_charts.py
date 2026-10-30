"""Save simple feature-importance charts for tree models as SVG files."""
from pathlib import Path
from html import escape
from sklearn.model_selection import train_test_split
from .pipeline import datasets, make_pipeline
from .config import RANDOM_STATE, TEST_SIZE, SPEND_COLUMNS
from .train import build_regressor, build_classifier

CHART_DIR = Path(__file__).resolve().parents[1] / "models" / "charts"

def save_bar_chart(filename, title, pairs, color):
    """Write a lightweight SVG horizontal bar chart without extra packages."""
    pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:12]
    width, row, left = 900, 29, 265
    height = 95 + row * len(pairs)
    largest = max([v for _, v in pairs] + [0.00001])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#151922"/>', f'<text x="24" y="38" fill="#e8eaf0" font-family="Arial" font-size="20" font-weight="bold">{escape(title)}</text>']
    for i, (name, value) in enumerate(pairs):
        y = 70 + i * row
        bar = max(1, (width - left - 50) * value / largest)
        parts += [f'<text x="24" y="{y+15}" fill="#aab0bf" font-family="Arial" font-size="12">{escape(str(name)[:36])}</text>', f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="18" rx="4" fill="{color}"/>', f'<text x="{left+bar+7:.1f}" y="{y+14}" fill="#e8eaf0" font-family="Arial" font-size="11">{value:.3f}</text>']
    parts.append('</svg>')
    (CHART_DIR / filename).write_text('\n'.join(parts), encoding='utf-8')

def generate_tree_charts():
    """Train tree estimators on the fixed training data and save importance charts."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    data, cls, reg = datasets()
    xr_train, _, yr_train, _ = train_test_split(reg, data.Total_Spend, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for name, model, color in [("decision_tree_regressor.svg", build_regressor("Decision Tree"), "#f5b942"), ("random_forest_regressor.svg", build_regressor("Random Forest"), "#2dd4bf")]:
        pipe = make_pipeline(xr_train, model).fit(xr_train, yr_train)
        names = pipe.named_steps["prep"].get_feature_names_out()
        save_bar_chart(name, name.replace('_', ' ').replace('.svg', ''), zip(names, pipe.named_steps["model"].feature_importances_), color)
    xc_train, _, yc_train, _ = train_test_split(cls, data.Response.astype(int), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data.Response)
    for name, model, color in [("decision_tree_classifier.svg", build_classifier("Decision Tree"), "#f5b942"), ("random_forest_classifier.svg", build_classifier("Random Forest"), "#2dd4bf")]:
        pipe = make_pipeline(xc_train, model).fit(xc_train, yc_train)
        names = pipe.named_steps["prep"].get_feature_names_out()
        save_bar_chart(name, name.replace('_', ' ').replace('.svg', ''), zip(names, pipe.named_steps["model"].feature_importances_), color)

if __name__ == "__main__":
    generate_tree_charts()
