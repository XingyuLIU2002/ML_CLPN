import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

def make_strata(y):
    return y

def train_randomforest_model(X, y, selected_vars, random_state=42, test_size=0.3, val_size=0.2):
    # 使用分层抽样
    strata = make_strata(y)
    X_temp, X_te, y_temp, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=strata
    )
    
    # 虽然RF不需要验证集来早停，但为了保持与主程序接口一致，需要生成它
    strata_temp = make_strata(y_temp)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state,
        stratify=strata_temp
    )
    
    cat_vars = [c for c in selected_vars if c in X.columns and str(X[c].dtype) == "category"]
    num_vars = [c for c in selected_vars if c in X.columns and c not in cat_vars]
    
    # 预处理
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
    
    # 基础模型
    base_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1
    )
    
    # 搜索空间：针对过拟合，更严格地限制深度和最小叶子样本数
    param_dist = {
        'rf__n_estimators': [300, 500, 800], 
        'rf__max_depth': [None, 8, 12, 16],
        'rf__min_samples_leaf': [5, 8, 12, 15],
        'rf__min_samples_split': [10, 15, 20], 
        'rf__max_features': ['sqrt', 0.2, 0.3, 0.4], # 探索更低的比例
        'rf__bootstrap': [True]
    }
    
    pipe = Pipeline([("prep", preprocess), ("rf", base_model)])
    
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )
    
    print(f"Start Random Forest search on {len(X_tr)} samples...")
    # 注意：只在训练集(X_tr)上进行搜索和训练
    rs.fit(X_tr[selected_vars], y_tr)
    print("Best RF Params:", rs.best_params_)
    
    # 获取最佳模型
    final_model = rs.best_estimator_
    
    pred_tr = final_model.predict_proba(X_tr[selected_vars])[:, 1]
    pred_val = final_model.predict_proba(X_val[selected_vars])[:, 1]
    pred_te = final_model.predict_proba(X_te[selected_vars])[:, 1]
    pred_tr_label = (pred_tr >= 0.5).astype(int)
    pred_val_label = (pred_val >= 0.5).astype(int)
    pred_te_label = (pred_te >= 0.5).astype(int)
    
    feature_names = final_model.named_steps['prep'].get_feature_names_out()
    
    return {
        'model': final_model,
        'preprocessor': final_model.named_steps['prep'],
        'feature_names': feature_names,
        'best_params': rs.best_params_,
        
        'X_train': X_tr, 
        'X_val': X_val,
        'X_test': X_te,
        'y_train': y_tr, 
        'y_val': y_val,
        'y_test': y_te,

        'train_auc': roc_auc_score(y_tr, pred_tr),
        'train_acc': accuracy_score(y_tr, pred_tr_label),
        'train_f1': f1_score(y_tr, pred_tr_label),
        'val_auc': roc_auc_score(y_val, pred_val),
        'val_acc': accuracy_score(y_val, pred_val_label),
        'val_f1': f1_score(y_val, pred_val_label),
        'test_auc': roc_auc_score(y_te, pred_te),
        'test_acc': accuracy_score(y_te, pred_te_label),
        'test_f1': f1_score(y_te, pred_te_label)
    }