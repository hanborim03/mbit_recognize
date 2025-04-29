from load_data import load_and_preprocess
from label_encoder import mbti_to_binary_labels
from text_encoder import get_bert_embeddings
from model import MBTILinearClassifier
from train import train

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

if __name__ == "__main__":
    # Step 1: 데이터 로드 및 전처리
    texts, labels = load_and_preprocess("data/mbti_1.csv")

    # Step 2: 각 축별 이진 라벨 생성
    binary_labels = mbti_to_binary_labels(labels)

    # Step 3: BERT 임베딩 추출 (한 번만!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = get_bert_embeddings(texts, device=device)

    # Step 4: 각 축별로 분류기 학습 및 평가
    for axis in ['IE', 'NS', 'TF', 'JP']:
        print(f"\n==== {axis} 이진 분류 ====")
        y = np.array(binary_labels[axis])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        model = MBTILinearClassifier(input_dim=X.shape[1], output_dim=2)
        train(model, X_train, y_train, X_test, y_test, class_weights, epochs=5)
