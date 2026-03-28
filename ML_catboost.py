import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from catboost import CatBoostClassifier


def make_strata(y):
    return y


def train_catboost_model(
    X, y, selected_vars,
    random_state=42,
    test_size=0.30,
    val_size=0.2,
    scoring="roc_auc"
):
    """
    CatBoost 二分类模型
    """

    # ========== 1. Train / Test split ==========
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=make_strata(y)
    )

    # ========== 2. 变量类型 ==========
    cat_vars = [c for c in selected_vars if c in X.columns and str(X[c].dtype) == "category"]
    num_vars = [c for c in selected_vars if c in X.columns and c not in cat_vars]

    # ========== 3. 预处理 ==========
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_vars),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_vars),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # ========== 4. 基础模型 ==========
    base_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False
    )

    pipe_cat = Pipeline([
        ("prep", preprocess),
        ("cat", base_model)
    ])

    # ========== 5. 超参数空间 ==========
    param_dist = {
        "cat__iterations": [500, 800, 1000],
        "cat__depth": [4, 6, 8],
        "cat__learning_rate": [0.01, 0.03, 0.05],
        "cat__l2_leaf_reg": [1, 3, 5, 7],
        "cat__subsample": [0.7, 0.8, 0.9],
        "cat__rsm": [0.7, 0.8, 0.9],
        "cat__scale_pos_weight": [1, 2, 3, 5]
    }

    # ========== 6. RandomizedSearchCV ==========
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    rs_cat = RandomizedSearchCV(
        estimator=pipe_cat,
        param_distributions=param_dist,
        n_iter=30,
        scoring=scoring,
        n_jobs=-1,
        cv=cv,
        refit=True,
        random_state=random_state,
        verbose=1
    )

    print("Start CatBoost hyperparameter search...")
    rs_cat.fit(X_tr[selected_vars], y_tr)

    print("\n=== CatBoost RandomizedSearchCV ===")
    print(f"Best CV AUC: {rs_cat.best_score_:.4f}")
    print("Best Params:", rs_cat.best_params_)

    # ========== 7. Train / Val split ==========
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr[selected_vars], y_tr,
        test_size=val_size,
        random_state=random_state,
        stratify=make_strata(y_tr)
    )

    final_preprocess = rs_cat.best_estimator_.named_steps["prep"]
    final_preprocess.fit(X_tr2, y_tr2)

    X_tr2_t = final_preprocess.transform(X_tr2)
    X_val_t = final_preprocess.transform(X_val)
    X_te_t = final_preprocess.transform(X_te[selected_vars])

    # ========== 8. Final model ==========
    best_params = {k.replace("cat__", ""): v for k, v in rs_cat.best_params_.items()}

    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=100,
        allow_writing_files=False,
        **best_params
    )

    if "iterations" in best_params:
        final_model.set_params(iterations=2000)

    print("\nTraining final CatBoost model with Early Stopping...")
    final_model.fit(
        X_tr2_t, y_tr2,
        eval_set=(X_val_t, y_val),
        early_stopping_rounds=50,
        use_best_model=True
    )

    # ========== 9. Predictions ==========
    pred_tr = final_model.predict_proba(X_tr2_t)[:, 1]
    pred_val = final_model.predict_proba(X_val_t)[:, 1]
    pred_te = final_model.predict_proba(X_te_t)[:, 1]

    return {
        "model": final_model,
        "preprocessor": final_preprocess,
        "feature_names": final_preprocess.get_feature_names_out(),

        "X_train": X_tr2_t,
        "X_val": X_val_t,
        "X_test": X_te_t,

        "y_train": y_tr2,
        "y_val": y_val,
        "y_test": y_te,

        "train_auc": roc_auc_score(y_tr2, pred_tr),
        "train_acc": accuracy_score(y_tr2, (pred_tr >= 0.5).astype(int)),
        "train_f1": f1_score(y_tr2, (pred_tr >= 0.5).astype(int)),

        "val_auc": roc_auc_score(y_val, pred_val),
        "val_acc": accuracy_score(y_val, (pred_val >= 0.5).astype(int)),
        "val_f1": f1_score(y_val, (pred_val >= 0.5).astype(int)),

        "test_auc": roc_auc_score(y_te, pred_te),
        "test_acc": accuracy_score(y_te, (pred_te >= 0.5).astype(int)),
        "test_f1": f1_score(y_te, (pred_te >= 0.5).astype(int))
    }