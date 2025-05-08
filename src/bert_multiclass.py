import torch
import os
import pandas as pd
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

# 0. GPU 사용 정보 표시
print("="*40)
if torch.cuda.is_available():
    print("✅ CUDA(GPU) 사용 가능!")
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} 이름: {torch.cuda.get_device_name(i)}")
    print(f"현재 선택된 GPU 번호: {torch.cuda.current_device()}")
    print(f"현재 선택된 GPU 이름: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("❌ CUDA(GPU) 사용 불가. CPU로 동작합니다.")
print("="*40)

# 1. 결과 저장 폴더 생성
RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

# 2. 데이터 로드 및 전처리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "../data/mbti_1.csv")
df = pd.read_csv(csv_path)
texts = df['posts'].apply(clean_text).tolist()
labels = df['type'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 3. 멀티클래스 분류 (16개 유형)
print("\n==== [멀티클래스] 16개 MBTI 유형 분류 ====")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, stratify=y, random_state=42)
train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
eval_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)
train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
columns = ['input_ids', 'attention_mask', 'label']
train_dataset.set_format(type='torch', columns=columns)
eval_dataset.set_format(type='torch', columns=columns)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
training_args = TrainingArguments(
    output_dir=os.path.join(RESULT_DIR, 'bert_multiclass'),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="none"
)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average="weighted")
    }
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
eval_result = trainer.evaluate()
print("[Multiclass] Eval result:", eval_result)

# 예측값 및 평가 결과 저장
y_pred = trainer.predict(eval_dataset).predictions.argmax(axis=1)
pd.DataFrame({
    "true_label": y_test,
    "pred_label": y_pred
}).to_csv(os.path.join(RESULT_DIR, "bert_multiclass_predictions.csv"), index=False)
with open(os.path.join(RESULT_DIR, "bert_multiclass_metrics.txt"), "w") as f:
    f.write(str(eval_result) + "\n")

# 4. 각 축별 이진 분류
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
    train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
    eval_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
    train_dataset.set_format(type='torch', columns=columns)
    eval_dataset.set_format(type='torch', columns=columns)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    training_args = TrainingArguments(
        output_dir=os.path.join(RESULT_DIR, f'bert_{axis}'),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    eval_result = trainer.evaluate()
    print(f"[{axis}] Eval result:", eval_result)

    # 예측값 및 평가 결과 저장
    y_pred = trainer.predict(eval_dataset).predictions.argmax(axis=1)
    pd.DataFrame({
        "true_label": y_test,
        "pred_label": y_pred
    }).to_csv(os.path.join(RESULT_DIR, f"bert_{axis}_predictions.csv"), index=False)
    with open(os.path.join(RESULT_DIR, f"bert_{axis}_metrics.txt"), "w") as f:
        f.write(str(eval_result) + "\n")

print("All results saved to result/ folder.")
