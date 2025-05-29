import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 디버깅용

import torch
import pandas as pd
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from collections import Counter
import pickle

print("="*40)
if torch.cuda.is_available():
    print(f"현재 선택된 GPU 이름: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA(GPU) 사용 불가. CPU로 동작합니다.")
print("="*40)

RESULT_DIR = "result/bert_multiclass"
os.makedirs(RESULT_DIR, exist_ok=True)

valid_mbti = [
    'INFJ','INFP','INTJ','INTP','ISTJ','ISTP','ISFJ','ISFP',
    'ENFJ','ENFP','ENTJ','ENTP','ESTJ','ESTP','ESFJ','ESFP'
]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "merged_data.csv")
df = pd.read_csv(csv_path, dtype=str)

# 결측값 처리 및 공백 제거, 대문자 통일
df['q_mbti'] = df['q_mbti'].fillna("").str.strip().str.upper()
df['a_mbti'] = df['a_mbti'].fillna("").str.strip().str.upper()
df['question'] = df['question'].fillna("").str.strip()
df['answer'] = df['answer'].fillna("").str.strip()

# 유효 데이터 필터링
df_q = df[df['q_mbti'].isin(valid_mbti) & df['question'].ne("")]
df_a = df[df['a_mbti'].isin(valid_mbti) & df['answer'].ne("")]

texts = pd.concat([
    df_q['question'].apply(clean_text),
    df_a['answer'].apply(clean_text)
]).tolist()
labels = pd.concat([
    df_q['q_mbti'],
    df_a['a_mbti']
]).tolist()

# 라벨 종류 및 분포 출력
print("라벨 종류:", set(labels))
print("라벨 개수:", len(set(labels)))
print("전체 데이터셋 크기:", len(labels))
print("클래스별 데이터 분포:")
print(Counter(labels))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print("인코딩된 라벨 min:", y.min(), "max:", y.max())
print("클래스 개수:", len(label_encoder.classes_))

# 모델과 토크나이저 선언
MODEL_NAME = 'klue/bert-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))

print("\n==== [멀티클래스] 16개 MBTI 유형 분류 ====")
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, stratify=y, random_state=42
)

print("Train 데이터 분포:", Counter(y_train))
print("Test 데이터 분포:", Counter(y_test))

train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
eval_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

columns = ['input_ids', 'attention_mask', 'label']
train_dataset.set_format(type='torch', columns=columns)
eval_dataset.set_format(type='torch', columns=columns)

training_args = TrainingArguments(
    output_dir=os.path.join(RESULT_DIR, 'bert_multiclass'),
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="none",
    learning_rate=3e-5
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {
        'accuracy': acc,
        'f1': f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print("학습 시작...")
trainer.train()

print("평가 중...")
eval_result = trainer.evaluate()
print("[Multiclass] Eval result:", eval_result)

y_pred = trainer.predict(eval_dataset).predictions.argmax(axis=1)
pd.DataFrame({
    "true_label": y_test,
    "pred_label": y_pred
}).to_csv(os.path.join(RESULT_DIR, "bert_multiclass_predictions.csv"), index=False)

with open(os.path.join(RESULT_DIR, "bert_multiclass_metrics.txt"), "w") as f:
    f.write(str(eval_result) + "\n")

# 모델과 토크나이저 저장
trainer.save_model(os.path.join(RESULT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(RESULT_DIR, "final_tokenizer"))

# 라벨 인코더 저장
with open(os.path.join(RESULT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# 4개 축별 이진 분류
def mbti_to_binary_labels(mbti_types):
    ie = [0 if t[0].upper() == 'I' else 1 for t in mbti_types]
    ns = [0 if t[1].upper() == 'N' else 1 for t in mbti_types]
    tf = [0 if t[2].upper() == 'T' else 1 for t in mbti_types]
    jp = [0 if t[3].upper() == 'J' else 1 for t in mbti_types]
    return {'IE': ie, 'NS': ns, 'TF': tf, 'JP': jp}

binary_labels = mbti_to_binary_labels(labels)

for axis in tqdm(['IE', 'NS', 'TF', 'JP'], desc="Binary Classification Progress"):
    print(f"\n==== [{axis}] 이진 분류 ====")
    y_bin = binary_labels[axis]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y_bin, test_size=0.2, stratify=y_bin, random_state=42
    )
    print(f"[{axis}] Train 데이터 분포:", Counter(y_train))
    print(f"[{axis}] Test 데이터 분포:", Counter(y_test))

    train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
    eval_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    train_dataset.set_format(type='torch', columns=columns)
    eval_dataset.set_format(type='torch', columns=columns)

    model_bin = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    training_args_bin = TrainingArguments(
        output_dir=os.path.join(RESULT_DIR, f'bert_{axis}'),
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
        learning_rate=3e-5
    )

    trainer_bin = Trainer(
        model=model_bin,
        args=training_args_bin,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer_bin.train()
    eval_result_bin = trainer_bin.evaluate()
    print(f"[{axis}] Eval result:", eval_result_bin)

    y_pred_bin = trainer_bin.predict(eval_dataset).predictions.argmax(axis=1)

    # 이진 분류 모델과 토크나이저 저장
    bin_model_dir = os.path.join(RESULT_DIR, f"final_model_{axis}")
    bin_tokenizer_dir = os.path.join(RESULT_DIR, f"final_tokenizer_{axis}")
    print(f"[{axis}] 모델 저장 위치: {bin_model_dir}")
    trainer_bin.save_model(bin_model_dir)
    print("  파일 목록:", os.listdir(bin_model_dir) if os.path.exists(bin_model_dir) else "폴더 없음")
    tokenizer.save_pretrained(bin_tokenizer_dir)
    print(f"[{axis}] 토크나이저 저장 위치: {bin_tokenizer_dir}")
    print("  파일 목록:", os.listdir(bin_tokenizer_dir) if os.path.exists(bin_tokenizer_dir) else "폴더 없음")

    # 예측 결과 저장
    pd.DataFrame({
        "true_label": y_test,
        "pred_label": y_pred_bin
    }).to_csv(os.path.join(RESULT_DIR, f"bert_{axis}_predictions.csv"), index=False)
    
    with open(os.path.join(RESULT_DIR, f"bert_{axis}_metrics.txt"), "w") as f:
        f.write(str(eval_result_bin) + "\n")

print("전체 학습 및 평가 완료")