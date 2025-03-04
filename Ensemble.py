import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score

# 데이터 로드
train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

X = train.drop(columns=['임신 성공 여부', 'PGD 시술 여부', '착상 전 유전 검사 사용 여부', '불임 원인 - 여성 요인', '난자 채취 경과일'])
y = train['임신 성공 여부']

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

# Ordinal Encoding 적용
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_encoded = X.copy()
X_train_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

X_test_encoded = test.reindex(columns=X_train_encoded.columns, fill_value=-1)
X_test_encoded[categorical_columns] = ordinal_encoder.transform(X_test_encoded[categorical_columns])

# 결측값 처리
X_train_encoded[numeric_columns] = X_train_encoded[numeric_columns].fillna(0)
X_test_encoded[numeric_columns] = X_test_encoded[numeric_columns].fillna(0)

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_train_encoded, y, test_size=0.2, random_state=42)

# 모델 설정
models = {
    'XGBoost': XGBClassifier(tree_method='gpu_hist', random_state=42),
    'CatBoost': CatBoostClassifier(verbose=False, random_state=42),
    'LightGBM': LGBMClassifier(verbose=-1, random_state=21),
    'TabNet': TabNetClassifier(verbose=0)
}

ensemble_results = []

# 가능한 모든 모델 조합을 앙상블
for r in range(1, len(models) + 1):
    for model_comb in combinations(models.keys(), r):
        print(f"Training ensemble: {model_comb}")

        preds = np.zeros(len(y_val))
        trained_models = []

        for model_name in model_comb:
            model = models[model_name]
            model.fit(np.array(X_train), y_train)
            preds += model.predict_proba(np.array(X_val))[:, 1]  # 확률값 누적
            trained_models.append(model)

        preds /= len(model_comb)  # 평균 확률값
        roc_auc = roc_auc_score(y_val, preds)

        ensemble_results.append([', '.join(model_comb), roc_auc, trained_models])
        print(f"Ensemble: {model_comb} -> ROC AUC: {roc_auc:.4f}")

# 결과를 엑셀 파일로 저장
ensemble_results_df = pd.DataFrame([[x[0], x[1]] for x in ensemble_results], columns=['Ensemble Models', 'ROC AUC'])
ensemble_results_df.to_excel('ensemble_results.xlsx', index=False)

print("Ensemble results saved to ensemble_results.xlsx")

#가장 높은 ROC AUC를 가진 모델 선택
best_ensemble = max(ensemble_results, key=lambda x: x[1])
best_models = best_ensemble[2]
print(f"Best ensemble: {best_ensemble[0]} with ROC AUC: {best_ensemble[1]:.4f}")

# 테스트 데이터 예측
X_test_np = np.array(X_test_encoded)
test_preds = np.zeros(len(X_test_np))

for model in best_models:
    test_preds += model.predict_proba(X_test_np)[:, 1]  # 확률값 누적

test_preds /= len(best_models)  # 평균 확률값

# 결과 저장
pd.DataFrame({'ID': test.index, 'Prediction': test_preds}).to_csv('best_model_predictions.csv', index=False)

print("Predictions saved to best_model_predictions.csv")
