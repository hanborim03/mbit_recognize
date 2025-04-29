import pandas as pd
import re
from sklearn.feature_extraction import text

# 영어 불용어
stop_words = text.ENGLISH_STOP_WORDS

def clean_text(text):
    # 1. 소문자 변환
    text = text.lower()
    
    # 2. 하이퍼링크 제거
    text = re.sub(r"http\S+|www.\S+", "", text)

    # 3. MBTI 단어 마스킹
    mbti_types = ['infj','infp','intj','intp','istj','istp','isfj','isfp',
                  'enfj','enfp','entj','entp','estj','estp','esfj','esfp']
    for mbti in mbti_types:
        text = text.replace(mbti, "mbti")

    # 4. 특수문자 제거
    text = re.sub(r"[^a-z\s]", "", text)

    # 5. 불용어 제거
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text

def load_and_preprocess(csv_path="data/mbti_1.csv"):
    df = pd.read_csv(csv_path)
    
    # 결측치 제거
    df.dropna(inplace=True)

    # 전처리 수행
    texts = df['posts'].apply(clean_text).tolist()
    labels = df['type'].tolist()

    return texts, labels
