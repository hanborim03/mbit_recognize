import re

def clean_text(text):
    # URL 제거
    text = re.sub(r"http\S+|www.\S+", "", str(text))

    # MBTI 유형 대체 (대소문자 모두 포함)
    mbti_types = [
        'INFJ','INFP','INTJ','INTP','ISTJ','ISTP','ISFJ','ISFP',
        'ENFJ','ENFP','ENTJ','ENTP','ESTJ','ESTP','ESFJ','ESFP'
    ]
    for mbti in mbti_types:
        text = re.sub(mbti, 'mbti', text, flags=re.IGNORECASE)

    # 특수문자 제거 (한글, 영어, 숫자, 공백만 허용)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

    # 공백 정규화
    text = re.sub(r"\s+", " ", text).strip()

    return text