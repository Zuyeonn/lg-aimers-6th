import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------
# 1) 데이터 불러오기
# ---------------------------
train_df = pd.read_csv("train.csv", encoding='utf-8')
test_df  = pd.read_csv("test.csv", encoding='utf-8')
sample_submission = pd.read_csv("C:/Users/pc/Downloads/LG_Aimers_Dacon/sample_submission.csv", encoding='utf-8')

# ---------------------------
# 2) 학습에서 제외할 컬럼 정의
# ---------------------------
# 표의 설명에 따라 모델 학습 시 사용하지 않을 컬럼
exclude_cols = [
    "난자 해동 경과일",
    "PGS 시술 여부",
    "PGD 시술 여부",
    "착상 전 유전 검사(PGS) 사용 여부",
    "착상 전 유전 진단(PGD) 사용 여부",
    "난자 기증자 나이",
    "정자 기증자 나이"
]

# ---------------------------
# 3) ID 및 타깃 컬럼 지정
# ---------------------------
id_col = "ID"
target_col = "임신 성공 여부"  # 표의 마지막 행에 명시된 타깃

# ---------------------------
# 4) Test 데이터 전처리: ID 분리 및 제외할 컬럼 제거
# ---------------------------
test_id = test_df[id_col].values
X_test = test_df.drop(id_col, axis=1)
drop_cols_test = [col for col in exclude_cols if col in X_test.columns]
X_test = X_test.drop(drop_cols_test, axis=1)

# ---------------------------
# 5) Train 데이터 전처리: Feature와 타깃 분리, ID 및 제외할 컬럼 제거
# ---------------------------
X = train_df.drop(target_col, axis=1)
y = train_df[target_col]
if id_col in X.columns:
    X = X.drop(id_col, axis=1)
drop_cols_train = [col for col in exclude_cols if col in X.columns]
X = X.drop(drop_cols_train, axis=1)

# ---------------------------
# 6) 범주형 변수 지정 및 타입 변환
# ---------------------------
# 표에서 범주형으로 명시된 컬럼들을 cat_features에 지정합니다.
cat_features = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "단일 배아 이식 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부"
]

# 학습 데이터(X)와 테스트 데이터(X_test)에서 해당 컬럼이 존재하면 문자열로 변환
for col in cat_features:
    if col in X.columns:
        X[col] = X[col].astype(str)
    if col in X_test.columns:
        X_test[col] = X_test[col].astype(str)

# ---------------------------
# 7) 데이터셋 분할 (Train/Validation)
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
for col in cat_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(str)
    if col in X_val.columns:
        X_val[col] = X_val[col].astype(str)

# ---------------------------
# 8) CatBoost 모델 학습
# ---------------------------
catboost_model = CatBoostClassifier(
    iterations=330,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    loss_function='Logloss',
    verbose=10
)

catboost_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# ---------------------------
# 9) Validation 예측 및 평가 (ROC-AUC, Accuracy)
# ---------------------------
cat_proba_val = catboost_model.predict_proba(X_val)[:, 1]  # 1번 클래스 확률
cat_pred_val = (cat_proba_val >= 0.5).astype(int)

acc = accuracy_score(y_val, cat_pred_val)
auc = roc_auc_score(y_val, cat_proba_val)
print("[Validation Metrics]")
print(f"CatBoost Accuracy: {acc:.4f} | AUC: {auc:.4f}")

# ---------------------------
# 10) Test 예측
# ---------------------------
cat_proba_test = catboost_model.predict_proba(X_test)[:, 1]

# ---------------------------
# 11) 제출 파일 생성 (ID, probability)
# ---------------------------
submission_df = pd.DataFrame({
    id_col: test_id,
    "probability": cat_proba_test  # 예측된 임신 성공 확률
})
submission_df.to_csv("submission_catboost.csv", index=False, encoding='utf-8')
print("submission_catboost.csv 파일 생성 완료!")
