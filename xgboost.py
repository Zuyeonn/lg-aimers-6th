import optuna
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def early_stopping_callback(eval_metric='logloss', patience=10):
    return [(X_val, y_val)], {'early_stopping_rounds': patience, 'eval_metric': eval_metric, 'verbose': True}

# 데이터 로드 및 전처리 (ID 컬럼 제거)
train = pd.read_csv("C:/Users/pc/Downloads/LG_Aimers_Dacon/train.csv").drop(columns=['ID'])
test = pd.read_csv("C:/Users/pc/Downloads/LG_Aimers_Dacon/test.csv").drop(columns=['ID'])

# 학습에 사용할 피처와 타깃 설정 (제외할 컬럼은 제거)
X = train.drop(columns=['임신 성공 여부', 'PGD 시술 여부', '착상 전 유전 검사 사용 여부', '불임 원인 - 여성 요인', '난자 채취 경과일'], axis=1)
y = train['임신 성공 여부']

categorical_columns = [
    "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형", "배란 자극 여부", "배란 유도 유형", 
    "단일 배아 이식 여부", "착상 전 유전 진단 사용 여부", "남성 주 불임 원인", "남성 부 불임 원인", 
    "여성 주 불임 원인", "여성 부 불임 원인", "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인", 
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애", "불임 원인 - 자궁경부 문제", 
    "불임 원인 - 자궁내막증", "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 운동성", 
    "불임 원인 - 정자 형태", "배아 생성 주요 이유", "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", 
    "DI 시술 횟수", "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", 
    "DI 출산 횟수", "난자 출처", "정자 출처", "난자 기증자 나이", "정자 기증자 나이", 
    "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부", "PGS 시술 여부"
]

numeric_columns = [
    "임신 시도 또는 마지막 임신 경과 연수", "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수", "이식된 배아 수", 
    "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수", 
    "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수", 
    "난자 해동 경과일", "난자 혼합 경과일", "배아 이식 경과일", "배아 해동 경과일"
]

# 범주형 변수는 문자열로 변환
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# OrdinalEncoder 적용
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_encoded = X.copy()
X_train_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
X_test_encoded = test.copy()
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# 수치형 결측치는 각 컬럼의 중앙값(median)으로 채움
for col in numeric_columns:
    median_val = X_train_encoded[col].median()
    X_train_encoded[col] = X_train_encoded[col].fillna(median_val)
    X_test_encoded[col] = X_test_encoded[col].fillna(median_val)

# Train/Validation Split (80:20, random_state 고정)
# 여기서는 튜닝을 위해 split한 데이터를 사용합니다.
X_train, X_val, y_train, y_val = train_test_split(X_train_encoded.values, y, test_size=0.2, random_state=42)

# Optuna 튜닝을 위한 결과 저장 리스트
study_results = []

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'tree_method': 'gpu_hist',
        'random_state': 42
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    study_results.append({**params, 'roc_auc': roc_auc})
    pd.DataFrame(study_results).to_excel('optuna_tuning_results.xlsx', index=False)
    print(f"Intermediate result saved to optuna_tuning_results.xlsx (Trial {len(study_results)})")
    return roc_auc

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)

print("Tuning results saved to optuna_tuning_results.xlsx")

# 최적의 파라미터와 최고 성능 출력 및 csv 파일 저장
best_params = study.best_params
best_trial_roc_auc = study.best_value
print("최적의 파라미터:", best_params)
print("최고의 ROC AUC (튜닝 시):", best_trial_roc_auc)

# 최종 모델을 전체 학습 데이터(전처리된 전체 training set)로 재학습
final_model = XGBClassifier(**best_params)
final_model.fit(X_train_encoded, y)  # X_train_encoded는 전체 training 데이터 (DataFrame)
final_pred_proba = final_model.predict_proba(X_train_encoded)[:, 1]
final_roc_auc = roc_auc_score(y, final_pred_proba)
print("최종 모델의 전체 training data ROC AUC:", final_roc_auc)

# 최종 결과를 csv 파일로 저장
final_results = {**best_params, "best_trial_roc_auc": best_trial_roc_auc, "final_roc_auc": final_roc_auc}
final_results_df = pd.DataFrame([final_results])
final_results_df.to_csv("final_xgboost_results.csv", index=False)
print("최종 결과가 final_xgboost_results.csv 파일로 저장되었습니다.")
