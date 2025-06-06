import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import os
import re

# 1. 불용어(stopwords) 리스트
STOPWORDS = {
    '이', '그', '저', '것', '분위기', '이미지', '사진', '장면', '모습', '장소',
    '있다', '있', '은', '는', '을', '를', '가', '의', '에', '로', '과', '와', '적', '하다', '되다', '설명', '이다', '같다', '수', '위', '고', '도', '으로', '에서', '에게', '께서', '부터', '까지', '보다', '처럼', '만큼', '만', '뿐', '밖에', '마다', '씩', '이나', '나', '이나마', '이나마도', '이나도', '이나마도', '이나마도', '이나마도', '이나마도'
}

# 2. 명사+형용사(감성 키워드) 추출 함수 (불용어 필터링 포함)
try:
    from konlpy.tag import Okt
    def extract_keywords(text, topk=5, stopwords=None):
        if stopwords is None:
            stopwords = set()
        okt = Okt()
        morphs = okt.pos(text, norm=True, stem=True)
        keywords = [word for word, pos in morphs if (pos == 'Noun' or pos == 'Adjective') and word not in stopwords]
        from collections import Counter
        counts = Counter(keywords)
        return [word for word, _ in counts.most_common(topk)]
except ImportError:
    def extract_keywords(text, topk=5, stopwords=None):
        if stopwords is None:
            stopwords = set()
        words = re.findall(r'[가-힣]{2,}(?:하다|스럽다|롭다|한|로운|적|스러운)?', text)
        filtered = [w for w in words if w and w not in stopwords]
        from collections import Counter
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(topk)]

# 3. 전체 MBTI(16종) 분류 예측 함수
def predict_mbti_multiclass(text, model_dir, tokenizer_dir, label_encoder_path, return_label_list=False):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        mbti_pred = label_encoder.inverse_transform([pred])[0]
        conf = probs[0, pred].item()
        prob_list = probs[0].tolist()
        if return_label_list:
            label_list = list(label_encoder.classes_)
            return mbti_pred, conf, prob_list, label_list
        else:
            return mbti_pred, conf, prob_list

# 4. 이진 분류 MBTI 예측 함수 (축별 확률 포함)
def predict_mbti_binary_with_probs(text):
    axes = ['IE', 'NS', 'TF', 'JP']
    axis_map = {
        'IE': ('I', 'E'),
        'NS': ('N', 'S'),
        'TF': ('T', 'F'),
        'JP': ('J', 'P')
    }
    mbti_binary_list = []
    conf_binary_list = []
    axis_probs = {}

    for axis in axes:
        bin_model_dir = os.path.abspath(f"result/bert_multiclass/final_model_{axis}")
        bin_tokenizer_dir = os.path.abspath(f"result/bert_multiclass/final_tokenizer_{axis}")
        bin_model = AutoModelForSequenceClassification.from_pretrained(bin_model_dir)
        bin_tokenizer = AutoTokenizer.from_pretrained(bin_tokenizer_dir)
        bin_inputs = bin_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        with torch.no_grad():
            bin_outputs = bin_model(**bin_inputs)
            bin_probs = torch.softmax(bin_outputs.logits, dim=1)
            bin_pred = torch.argmax(bin_probs, dim=1).item()
            mbti_binary_list.append(axis_map[axis][bin_pred])
            conf_binary_list.append(bin_probs[0, bin_pred].item())
            axis_probs[axis] = (
                bin_probs[0,0].item(),  # 첫 번째 클래스 확률
                bin_probs[0,1].item(),  # 두 번째 클래스 확률
                axis_map[axis][bin_pred],  # 예측 축
                bin_probs[0, bin_pred].item()  # 신뢰도
            )
    mbti_binary = ''.join(mbti_binary_list)
    avg_conf = sum(conf_binary_list) / len(conf_binary_list)
    return mbti_binary, avg_conf, axis_probs

# 5. 일반화된 생성문 후처리 함수
def remove_repeated_phrases(text, min_len=2):
    tokens = text.split()
    result = []
    prev = None
    for t in tokens:
        if prev is None or t != prev or len(t) < min_len:
            result.append(t)
        prev = t
    return ' '.join(result)

def clean_punctuation(text):
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\.\s+', '. ', text)
    text = re.sub(r'[^\w\s.,?!가-힣]', '', text)
    return text.strip()

def filter_short_sentences(text, min_len=5):
    sentences = re.split(r'[.?!]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_len]
    return '. '.join(sentences) + '.' if sentences else ''

def clean_generated_description(text):
    text = remove_repeated_phrases(text)
    text = clean_punctuation(text)
    text = filter_short_sentences(text)
    return text

# 6. 멀티모달 이미지 설명 생성 (생략: 기존 함수 그대로 사용)
from model import load_model_and_tokenizer, SimpleMultimodalModel
from PIL import Image
def generate_image_description(image_path, question, model_ckpt="my_multimodal_model_best.pt", model_name="EleutherAI/polyglot-ko-1.3b", tokenizer_dir="my_tokenizer_dir"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_name)
    if os.path.exists(model_ckpt):
        model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model = model.to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = model.clip_preprocess(image).unsqueeze(0).to(device)
    prompt = f"<image> {question}" if "<image>" not in question else question
    max_len = 32
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=max_len
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_token_pos_tensor = (input_ids[0] == image_token_id).nonzero(as_tuple=True)
    if len(image_token_pos_tensor[0]) == 0:
        print("[경고] 입력 프롬프트에 <image> 토큰이 없습니다!")
        return ""
    image_token_pos = image_token_pos_tensor[0][0].item()
    with torch.no_grad():
        outputs = model.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.2
        )
        description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description.strip()
