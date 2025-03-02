import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression  

# =======================================
# XGBoost 전처리 (수치형 결측치는 중앙값 처리)
# =======================================
def preprocess_xgb():
    print("XGBoost 전처리 시작...")
    
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    train.drop(columns=['ID'], inplace=True)
    test.drop(columns=['ID'], inplace=True)
    
    # 학습 데이터에서 제외 컬럼
    drop_columns = ['임신 성공 여부', 'PGD 시술 여부', '착상 전 유전 검사 사용 여부', '불임 원인 - 여성 요인', '난자 채취 경과일']
    X = train.drop(columns=drop_columns)
    y = train['임신 성공 여부']
    
    # 테스트 데이터를 학습 데이터와 동일한 피처 집합으로 재정렬 (없는 컬럼은 -1로 채움)
    test = test.reindex(columns=X.columns, fill_value=-1)
    
    categorical_columns = [
        "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형", "배란 자극 여부", "배란 유도 유형", 
        "단일 배아 이식 여부", "착상 전 유전 진단 사용 여부", "남성 주 불임 원인", "남성 부 불임 원인", 
        "여성 주 불임 원인", "여성 부 불임 원인", "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인", 
        "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애", "불임 원인 - 자궁경부 문제", 
        "불임 원인 - 자궁내막증", "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 운동성", 
        "불임 원인 - 정자 형태", "배아 생성 주요 이유", "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", 
        "DI 시술 횟수", "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", 
        "DI 출산 횟수", "난자 출처", "정자 출처", "난자 기증자 나이", "정자 기증자 나이", "동결 배아 사용 여부", 
        "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부", "PGS 시술 여부"
    ]
    numeric_columns = [
        "임신 시도 또는 마지막 임신 경과 연수", "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수", 
        "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수", 
        "해동 난자 수", "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", 
        "기증자 정자와 혼합된 난자 수", "난자 해동 경과일", "난자 혼합 경과일", "배아 이식 경과일", "배아 해동 경과일"
    ]
    
    # 범주형 변수: 문자열 변환 후 Ordinal Encoding
    for col in categorical_columns:
        X[col] = X[col].astype(str)
        test[col] = test[col].astype(str)
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
    test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])
    
    # 수치형 결측치는 0 대신 각 컬럼의 중앙값으로 채움
    for col in numeric_columns:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("XGBoost 전처리 완료")
    return X_train, X_val, y_train, y_val, test
    
    
# ======================================
#  CatBoost 전처리 (수치형 결측치는 중앙값 처리)
# ======================================
def preprocess_cat():
    print("CatBoost 전처리 시작...")
    train_data = pd.read_csv("train.csv", encoding='utf-8')
    test_data = pd.read_csv("test.csv", encoding='utf-8')
    sample_submission = pd.read_csv("sample_submission.csv", encoding='utf-8')
    
    # 학습에 제외할 컬럼 설정
    exclude_cols = [
        "난자 해동 경과일",
        "PGS 시술 여부",
        "PGD 시술 여부",
        "착상 전 유전 검사(PGS) 사용 여부",
        "착상 전 유전 진단(PGD) 사용 여부",
        "난자 기증자 나이",
        "정자 기증자 나이"
    ]
    id_col = "ID"
    target_col = "임신 성공 여부"
    
    # 테스트 데이터: ID 보존 후 제외할 컬럼 제거
    test_ids = test_data[id_col].values
    X_test = test_data.drop(id_col, axis=1)
    drop_cols_test = [col for col in exclude_cols if col in X_test.columns]
    X_test = X_test.drop(drop_cols_test, axis=1)
    
    # 학습 데이터: 타깃 및 ID, 제외할 컬럼 제거
    X = train_data.drop(target_col, axis=1)
    y = train_data[target_col]
    if id_col in X.columns:
        X = X.drop(id_col, axis=1)
    drop_cols_train = [col for col in exclude_cols if col in X.columns]
    X = X.drop(drop_cols_train, axis=1)
    
    # CatBoost에서 사용할 범주형 변수 리스트
    cat_features = [
        "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형", "배란 자극 여부", "배란 유도 유형", 
        "단일 배아 이식 여부", "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인", 
        "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인", "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", 
        "불임 원인 - 배란 장애", "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증", 
        "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태", 
        "배아 생성 주요 이유", "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수", 
        "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수", 
        "난자 출처", "정자 출처", "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부"
    ]
    
    # 범주형 변수: 문자열 변환
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(str)
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)
    
    # 수치형 컬럼: 결측치는 각 컬럼의 중앙값으로 대체
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(median_val)
    
    # 범주형 컬럼: 결측치는 "missing" 처리
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna("missing")
    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            X_test[col] = X_test[col].fillna("missing")
    
    # Train/Validation Split (80:20, random_state 고정)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # 학습 및 검증 데이터의 범주형 변수 타입 보장
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
        if col in X_val.columns:
            X_val[col] = X_val[col].astype(str)
    
    print("CatBoost 전처리 완료.")
    return X_train, X_val, y_train, y_val, X_test, test_ids, cat_features
    


# =====================
#  Stacking 
# =====================

# 데이터 전처리 실행
X_train_xgb, X_valid_xgb, y_train, y_valid, X_test_xgb = preprocess_xgb()
X_train_cat, X_valid_cat, y_train_cat, y_valid_cat, X_test_cat, test_ids, cat_features = preprocess_cat()

# 개별 모델 학습 (XGBoost & CatBoost)
print("XGBoost 모델 학습 중...")
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, 
                          subsample=0.8, colsample_bytree=0.8, gamma=0.1, 
                          reg_lambda=1.0, reg_alpha=0.5, eval_metric="logloss", 
                          random_state=42)
xgb_model.fit(X_train_xgb, y_train)
xgb_valid_preds = xgb_model.predict_proba(X_valid_xgb)[:, 1]
xgb_test_preds = xgb_model.predict_proba(X_test_xgb)[:, 1]

print("CatBoost 모델 학습 중...")
cat_model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, 
                               l2_leaf_reg=3.0, random_seed=42, verbose=0)
cat_model.fit(X_train_cat, y_train, cat_features=cat_features)
cat_valid_preds = cat_model.predict_proba(X_valid_cat)[:, 1]
cat_test_preds = cat_model.predict_proba(X_test_cat)[:, 1]

# Meta Model 학습 (Logistic Regression)
train_meta = np.column_stack((xgb_valid_preds, cat_valid_preds))
test_meta = np.column_stack((xgb_test_preds, cat_test_preds))

print("메타 모델(Logistic Regression) 학습 중...")
meta_model = LogisticRegression()  
meta_model.fit(train_meta, y_valid)
meta_valid_proba = meta_model.predict_proba(train_meta)[:, 1]
meta_test_proba = meta_model.predict_proba(test_meta)[:, 1]  # 테스트 데이터 예측 추가

# 평가
xgb_roc_auc = roc_auc_score(y_valid, xgb_valid_preds)
cat_roc_auc = roc_auc_score(y_valid, cat_valid_preds)
meta_roc_auc = roc_auc_score(y_valid, meta_valid_proba)

print(f"🔹 XGBoost ROC-AUC Score: {xgb_roc_auc:.4f}")
print(f"🔹 CatBoost ROC-AUC Score: {cat_roc_auc:.4f}")
print(f"✅ Meta Model (Stacking - Logistic Regression) ROC-AUC Score: {meta_roc_auc:.4f}")

# 테스트 예측 및 저장
submission = pd.DataFrame({
    "ID": test_ids,  # 기존 테스트 데이터에서 ID 사용
    "probability": meta_test_proba  # 메타 모델의 확률값을 예측값으로 저장
})
submission.to_csv("submission_stacking_logreg.csv", index=False, encoding='utf-8')

print("✅ 제출 완료: submission_stacking_logreg.csv")
