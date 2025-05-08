import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from preprocess import clean_text
from tqdm import tqdm
import joblib

# 결과 저장 폴더 생성
RESULT_DIR = "result/bow_gb"
os.makedirs(RESULT_DIR, exist_ok=True)

# 1. 데이터 로드 및 전처리
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "mbti_1.csv")
df = pd.read_csv(csv_path)
labels = df['type'].tolist()
texts = df['posts'].apply(clean_text).tolist()

# 2. 벡터화
vectorizer = CountVectorizer(max_df=0.57, min_df=0.09)
X = vectorizer.fit_transform(texts)

# 3. 데이터 분할 (멀티클래스)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, stratify=labels, random_state=42
)

# 4. 멀티클래스 분류 - Gradient Boosting + 진행률 + best model 저장
n_estimators = 100
clf = GradientBoostingClassifier(n_estimators=1, warm_start=True)
best_f1 = 0
best_model_path = os.path.join(RESULT_DIR, "best_gb_model.joblib")

for i in tqdm(range(1, n_estimators + 1), desc="Multiclass Gradient Boosting Progress"):
    clf.n_estimators = i
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    current_f1 = f1_score(y_test, y_pred, average='weighted')
    if current_f1 > best_f1:
        best_f1 = current_f1
        joblib.dump(clf, best_model_path)

print(f"[Multiclass] Best F1 Score: {best_f1:.4f}")
print(f"[Multiclass] Best model saved to {best_model_path}")

# 멀티클래스 예측 및 결과 저장
y_pred = clf.predict(X_test)
results_df = pd.DataFrame({
    "true_label": y_test,
    "pred_label": y_pred
})
results_df.to_csv(os.path.join(RESULT_DIR, "gb_multiclass_predictions.csv"), index=False)

with open(os.path.join(RESULT_DIR, "gb_multiclass_metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}\n")

print("[Multiclass] Prediction and metrics saved to result/")

# 5. 각 축별 이진 분류 (체크포인트 저장 추가)
def mbti_to_binary_labels(mbti_types):
    ie = [0 if t[0].upper() == 'I' else 1 for t in mbti_types]
    ns = [0 if t[1].upper() == 'N' else 1 for t in mbti_types]
    tf = [0 if t[2].upper() == 'T' else 1 for t in mbti_types]
    jp = [0 if t[3].upper() == 'J' else 1 for t in mbti_types]
    return {'IE': ie, 'NS': ns, 'TF': tf, 'JP': jp}

binary_labels = mbti_to_binary_labels(labels)

for axis in ['IE', 'NS', 'TF', 'JP']:
    print(f"\n==== [{axis}] BoW + GradientBoosting 이진 분류 ====")
    y_bin = binary_labels[axis]
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X, y_bin, test_size=0.2, stratify=y_bin, random_state=42
    )
    # 체크포인트 저장을 위한 구조
    clf_bin = GradientBoostingClassifier(n_estimators=1, warm_start=True)
    best_f1_bin = 0
    best_model_path_bin = os.path.join(RESULT_DIR, f"best_gb_{axis}_model.joblib")
    for i in tqdm(range(1, n_estimators + 1), desc=f"{axis} Gradient Boosting Progress"):
        clf_bin.n_estimators = i
        clf_bin.fit(X_train_bin, y_train_bin)
        y_pred_bin = clf_bin.predict(X_test_bin)
        current_f1_bin = f1_score(y_test_bin, y_pred_bin, average="weighted")
        if current_f1_bin > best_f1_bin:
            best_f1_bin = current_f1_bin
            joblib.dump(clf_bin, best_model_path_bin)
    print(f"[{axis}] Best F1 Score: {best_f1_bin:.4f}")
    print(f"[{axis}] Best model saved to {best_model_path_bin}")

    # 결과 저장 (최종 모델 기준)
    y_pred_bin = clf_bin.predict(X_test_bin)
    pd.DataFrame({
        "true_label": y_test_bin,
        "pred_label": y_pred_bin
    }).to_csv(os.path.join(RESULT_DIR, f"gb_{axis}_predictions.csv"), index=False)
    with open(os.path.join(RESULT_DIR, f"gb_{axis}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test_bin, y_pred_bin)}\n")
        f.write(f"F1 Score: {f1_score(y_test_bin, y_pred_bin, average='weighted')}\n")

print("All results saved to result/ folder.")
