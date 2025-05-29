import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import pandas as pd
from collections import Counter
import os

# 데이터 경로
csv_path = '/home/hanborim/mbti_project/data/merged_data.csv'

# 데이터 불러오기 및 라벨 분포 확인
df = pd.read_csv(csv_path)
labels = df['a_mbti'].dropna().tolist()
print("전체 MBTI 분포:", Counter(labels))
ie = [t[0].upper() for t in labels if isinstance(t, str) and len(t) == 4]
ns = [t[1].upper() for t in labels if isinstance(t, str) and len(t) == 4]
tf = [t[2].upper() for t in labels if isinstance(t, str) and len(t) == 4]
jp = [t[3].upper() for t in labels if isinstance(t, str) and len(t) == 4]
print("IE 분포:", Counter(ie))
print("NS 분포:", Counter(ns))
print("TF 분포:", Counter(tf))
print("JP 분포:", Counter(jp))

# 전체 MBTI(16종) 분류 모델 준비 (절대경로)
MODEL_DIR = os.path.abspath("result/bert_multiclass/final_model")
TOKENIZER_DIR = os.path.abspath("result/bert_multiclass/final_tokenizer")
LABEL_ENCODER_PATH = os.path.abspath("result/bert_multiclass/label_encoder.pkl")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
model.eval()


# 이진 분류 모델 경로 및 로드 (한 번만, 절대경로)
BINARY_AXES = ['IE', 'NS', 'TF', 'JP']
bin_models = {}
bin_tokenizers = {}
for axis in BINARY_AXES:
    bin_model_dir = os.path.abspath(f"result/bert_multiclass/final_model_{axis}")
    bin_tokenizer_dir = os.path.abspath(f"result/bert_multiclass/final_tokenizer_{axis}")
    bin_models[axis] = AutoModelForSequenceClassification.from_pretrained(bin_model_dir)
    bin_tokenizers[axis] = AutoTokenizer.from_pretrained(bin_tokenizer_dir)
    bin_models[axis].eval()

def predict_mbti(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        mbti_pred = label_encoder.inverse_transform([pred])[0]
        conf = probs[0, pred].item()
    return mbti_pred, conf

def predict_binary(text):
    binary_result = {}
    for axis in BINARY_AXES:
        bin_model = bin_models[axis]
        bin_tokenizer = bin_tokenizers[axis]
        inputs = bin_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        with torch.no_grad():
            outputs = bin_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()
        # I/E, N/S, T/F, J/P 매핑
        axis_map = {
            'IE': ('I', 'E'),
            'NS': ('N', 'S'),
            'TF': ('T', 'F'),
            'JP': ('J', 'P')
        }
        binary_result[axis] = {
            'pred': axis_map[axis][pred],
            'conf': conf,
            'probs': probs.tolist()[0]
        }
    return binary_result
    

if __name__ == "__main__":
    texts = [
        "나는 사람들과 함께 있는 게 너무 좋아.",
        "나는 혼자 있는 걸 좋아해.",
        "계획 세우는 걸 좋아하고, 감정보다 논리를 중시해.",
        "새로운 가능성을 상상하는 걸 즐긴다."
    ]
    for text in texts:
        print(f"\n입력: {text}")
        # 전체 MBTI 예측
        mbti, conf = predict_mbti(text)
        print(f"[전체 MBTI] 예측: {mbti} (신뢰도: {conf:.4f})")
        
        # 이진 분류 예측
        binary = predict_binary(text)
        bin_str = ''.join([binary[axis]['pred'] for axis in BINARY_AXES])
        print(f"[이진 분류 MBTI] 예측: {bin_str}")
        for axis in BINARY_AXES:
            probs = binary[axis]['probs']
            print(f"  {axis}: {probs[0]:.4f} / {probs[1]:.4f} -> {binary[axis]['pred']} (신뢰도: {binary[axis]['conf']:.4f})")
