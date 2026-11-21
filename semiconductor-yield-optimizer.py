import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_recall_curve,
    auc
)

warnings.filterwarnings("ignore")
start_time = time.time()

# ==========================================
# 1. DATA INGESTION & QUALITY AUDIT
# ==========================================
file_path = "uci-secom.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=['Time', 'Pass/Fail'], errors='ignore').copy()
X.columns = [f"Sensor_{i}" for i in range(X.shape[1])]
y = df['Pass/Fail'].map({-1: 0, 1: 1})  # 0 = Good Die/Wafer, 1 = Defective Wafer

# Drop sensors with >40% missing readings
missing_ratio = X.isnull().mean()
X_cleaned = X[missing_ratio[missing_ratio <= 0.40].index]

X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.2, random_state=42, stratify=y
)

print(f"=== FAB DATA AUDIT ===")
print(f"Total Wafers: {len(y)} | Baseline Yield: {(1 - y.mean())*100:.2f}% | Defect Rate: {y.mean()*100:.2f}%")
print(f"Active Sensors Retained: {X_cleaned.shape} / {X.shape}\n")

# ==========================================
# 2. MULTI-MODEL BENCHMARKING (STRATIFIED CV)
# ==========================================
print("=== BENCHMARKING CANDIDATE ARCHITECTURES (5-FOLD CV) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression (L2)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
}

for name, clf in models.items():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler()),
        ('anova_select', SelectKBest(score_func=f_classif, k=40)),
        ('classifier', clf)
    ])
    roc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"{name:<26} | Mean ROC-AUC: {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")

# ==========================================
# 3. EFFICIENT HYPERPARAMETER OPTIMIZATION
# ==========================================
print("\n=== TUNING CHAMPION PIPELINE (RANDOM FOREST + ANOVA) ===")
champion_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('var_thresh', VarianceThreshold(threshold=0.0)),
    ('scaler', StandardScaler()),
    ('anova_select', SelectKBest(score_func=f_classif)),
    ('classifier', RandomForestClassifier(
        class_weight='balanced',
        min_samples_leaf=4,     # Prevents overfitting on noisy fab outliers
        random_state=42,
        n_jobs=-1
    ))
])

# Focused, high-efficiency parameter grid
param_grid = {
    'anova_select__k':[20,40,60],
    'classifier__n_estimators':[100,200],
    'classifier__max_depth': [5, 8]
}

grid_search = GridSearchCV(
    estimator=champion_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Optimal Parameters: {grid_search.best_params_}")

# ==========================================
# 4. SEMICONDUCTOR COST-SENSITIVE THRESHOLD TUNING
# ==========================================
# Business Rule: Shipping a defective chip (False Negative) costs 10x more
# than re-testing a good wafer (False Positive).
COST_TEST_ESCAPE = 10   # Cost per missed defective wafer (FN)
COST_OVERKILL = 1       # Cost per false alarm / re-test (FP)

y_probs = best_model.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.20, 0.70, 51)
best_thresh = 0.50
lowest_fab_cost = float('inf')

for t in thresholds:
    preds = (y_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    total_cost = (fn * COST_TEST_ESCAPE) + (fp * COST_OVERKILL)
    if total_cost < lowest_fab_cost:
        lowest_fab_cost = total_cost
        best_thresh = t

# Compare Default (0.50) vs. Cost-Optimized Threshold
y_pred_default = (y_probs >= 0.50).astype(int)
y_pred_opt = (y_probs >= best_thresh).astype(int)

tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, y_pred_default).ravel()
default_cost = (fn_d * COST_TEST_ESCAPE) + (fp_d * COST_OVERKILL)

tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_test, y_pred_default).ravel()

print(f"\n=== FAB COST & YIELD OPTIMIZATION (TEST SET) ===")
print(f"Default Cutoff (0.50)   -> Defective Wafers Caught: {tp_d}/{tp_d+fn_d} | Estimated Fab Penalty Cost: {default_cost} units")
print(f"Optimized Cutoff ({best_thresh:.2f}) -> Defective Wafers Caught: {confusion_matrix(y_test, y_pred_opt).ravel()}/{tp_d+fn_d} | Estimated Fab Penalty Cost: {lowest_fab_cost} units")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f} | Execution Time: {time.time() - start_time:.2f}s\n")

print("--- Cost-Optimized Classification Report ---")
print(classification_report(y_test, y_pred_opt, target_names=["Good Wafer (0)", "Defective Wafer (1)"]))

# ==========================================
# 5. ROOT-CAUSE SENSOR EXTRACTION & 3-PANEL DASHBOARD
# ==========================================
var_mask = best_model.named_steps['var_thresh'].get_support()
anova_mask = best_model.named_steps['anova_select'].get_support()
selected_sensors = X_cleaned.columns[var_mask][anova_mask]

importances = best_model.named_steps['classifier'].feature_importances_
sensor_ranking = pd.Series(importances, index=selected_sensors).sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel 1: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
axes[0].plot(recall, precision, color='navy', lw=2, label=f'PR-AUC = {pr_auc:.3f}')
axes[0].axhline(y.mean(), color='gray', linestyle='--', label='Baseline Defect Rate (6.6%)')
axes[0].set_title("Precision-Recall Trade-off Curve")
axes[0].set_xlabel("Defect Recall (Sensitivity)")
axes[0].set_ylabel("Precision")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

# Panel 2: Cost-Optimized Confusion Matrix
cm_opt = confusion_matrix(y_test, y_pred_opt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_opt, display_labels=["Good", "Defect"])
disp.plot(ax=axes[1], cmap='Blues', colorbar=False)
axes[1].set_title(f"Cost-Optimized Matrix (Cutoff = {best_thresh:.2f})")

# Panel 3: Top 10 Root-Cause Sensors
sensor_ranking.head(10).sort_values().plot(kind='barh', ax=axes[2], color='#008080')
axes[2].set_title("Top 10 Root-Cause Fab Sensors")
axes[2].set_xlabel("Gini Feature Importance")

plt.tight_layout()
output_img = r"C:\Users\KARTIKE\Desktop\fab_yield_dashboard.png"
plt.savefig(output_img, dpi=300)
print(f"Saved 3-panel engineering dashboard to: {output_img}")
plt.show()
