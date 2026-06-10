"""
model.py — Обучение моделей и сохранение весов в data/model_weights.mw
Проект: Анализ популярности Принцесс Диснея
"""

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ─── Пути и подготовка ────────────────────────────────────────────────────────
DATA_DIR     = "data"
WEIGHTS_PATH = os.path.join(DATA_DIR, "model_weights.mw")
CSV_PATH     = "disney_princess_popularity_dataset_300_rows.csv"

os.makedirs(DATA_DIR, exist_ok=True)

# ─── Загрузка данных ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Загружено строк: {len(df)}, признаков: {df.shape[1]}")

# Исключаем неинформативные текстовые поля
USELESS = ["PrincessName", "FirstMovieTitle", "FirstMovieYear", "VillainName", "Top3Hashtags"]
df_clean = df.drop(columns=USELESS, errors="ignore")

# Разделяем признаки по типам
num_cols = df_clean.select_dtypes("number").columns.tolist()
cat_cols = df_clean.select_dtypes(include=["object", "string"]).columns.tolist()

# ─── Базовая предобработка (Pipeline) ────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), [c for c in num_cols if c not in ["PopularityScore", "IsIconic"]]),
    
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), [c for c in cat_cols if c not in ["PopularityScore", "IsIconic"]])
], remainder="drop")

# ════════════════════════════════════════════════════════
#  1. РЕГРЕССИЯ — PopularityScore
# ════════════════════════════════════════════════════════
print("\n── Обучение модели регрессии (Ridge) ──")
X_reg = df_clean.drop(columns=["PopularityScore", "IsIconic"], errors="ignore")
y_reg = df_clean["PopularityScore"]

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = Pipeline([("pre", preprocessor), ("model", Ridge(alpha=10.0))]) # Чуть увеличили альфа для регуляризации
reg_model.fit(X_tr_r, y_tr_r)

# Статистический baseline (предсказание средним значением)
y_baseline_pred = np.full_like(y_te_r, y_tr_r.mean())
r2_baseline = r2_score(y_te_r, y_baseline_pred)

r2_train = r2_score(y_tr_r, reg_model.predict(X_tr_r))
r2_test  = r2_score(y_te_r, reg_model.predict(X_te_r))
cv_r2    = cross_val_score(reg_model, X_reg, y_reg, cv=5, scoring="r2").mean()

print(f"  R² train: {r2_train:.4f} | R² test: {r2_test:.4f} | R² CV-5: {cv_r2:.4f}")
print(f"  [Статистика] Baseline R² (по среднему): {r2_baseline:.4f}")

# ════════════════════════════════════════════════════════
#  2. КЛАССИФИКАЦИЯ — IsIconic
# ════════════════════════════════════════════════════════
print("\n── Обучение модели классификации (Random Forest) ──")
y_clf = df_clean["IsIconic"]

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_reg, y_clf, test_size=0.2, random_state=42)

# Упростим параметры дерева, чтобы избежать переобучения на шуме
clf_model = Pipeline([("pre", preprocessor), ("model", RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))])
clf_model.fit(X_tr_c, y_tr_c)

acc_train = accuracy_score(y_tr_c, clf_model.predict(X_tr_c))
acc_test  = accuracy_score(y_te_c, clf_model.predict(X_te_c))
cv_acc    = cross_val_score(clf_model, X_reg, y_clf, cv=5, scoring="accuracy").mean()

# Baseline для классификации (доля самого частого класса)
majority_class_share = (y_clf == y_clf.mode()[0]).mean()
print(f"  Acc train: {acc_train:.4f} | Acc test: {acc_test:.4f} | Acc CV-5: {cv_acc:.4f}")
print(f"  [Статистика] Baseline Accuracy (по большинству): {majority_class_share:.4f}")

# ─── Сохранение данных для интерфейса ────────────────────────────────────────
payload = {
    "reg_model": reg_model,
    "clf_model": clf_model,
    "metrics": {
        "reg": {"train": r2_train, "test": r2_test, "cv": cv_r2, "baseline": r2_baseline},
        "clf": {"train": acc_train, "test": acc_test, "cv": cv_acc, "baseline": majority_class_share}
    },
    "data_summary": {
        "mean_pop": y_reg.mean(),
        "share_iconic": (y_clf == "Yes").mean()
    }
}

with open(WEIGHTS_PATH, "wb") as f:
    pickle.dump(payload, f)
print(f"\n✓ Артефакты успешно сохранены в {WEIGHTS_PATH}")