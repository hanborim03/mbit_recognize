import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    mbti_types = ['infj','infp','intj','intp','istj','istp','isfj','isfp',
                  'enfj','enfp','entj','entp','estj','estp','esfj','esfp']
    for mbti in mbti_types:
        text = text.replace(mbti, "mbti")
    text = re.sub(r"[^a-z\s]", "", text)
    return text