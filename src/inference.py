import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import SimpleMultimodalModel

def inference(image_path, question):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./my_multimodal_model_best"
    TOKENIZER_PATH = "./my_tokenizer_dir"

    # 토크나이저, 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = SimpleMultimodalModel()
    model.llm = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    model = model.to(device)
    model.eval()

    # 이미지 전처리
    image = Image.open(image_path).convert("RGB")
    image_tensor = model.clip_preprocess(image).unsqueeze(0).to(device)

    # 프롬프트에 <image> 토큰 포함
    prompt = f"<image> {question}" if "<image>" not in question else question

    # 토크나이즈
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

    # <image> 토큰 위치 찾기
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_token_pos_tensor = (input_ids[0] == image_token_id).nonzero(as_tuple=True)
    if len(image_token_pos_tensor[0]) == 0:
        print("[경고] 입력 프롬프트에 <image> 토큰이 없습니다!")
        return
    image_token_pos = image_token_pos_tensor[0][0].item()

    # 멀티모달 모델 추론 (forward 직접 호출, logits 생성)
    with torch.no_grad():
        # 멀티모달 임베딩 반영을 위해 forward 호출 (실제 LLaVA/KoLLaVA 구조)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=image_tensor,
            image_token_pos=torch.tensor([image_token_pos], dtype=torch.long)
        )
        # 오토리그레시브 생성 (더 자연스러운 답변 유도)
        generated = model.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=True,             # 샘플링 활성화
            top_p=0.95,                 # nucleus sampling
            temperature=0.8,            # 다양성 증가
            repetition_penalty=1.2      # 반복 억제
        )
        print("생성된 설명:", tokenizer.decode(generated[0], skip_special_tokens=True))

if __name__ == "__main__":
    image_path = "/home/hanborim/mbti_project/picture.jpg"
    question = "이 이미지의 분위기를 설명해줘"
    inference(image_path, question)
