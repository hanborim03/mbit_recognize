from torch.utils.data import Dataset
from PIL import Image
import torch

class VisualInstructDataset(Dataset):
    def __init__(self, data, coco_path, tokenizer, transform=None, min_label_tokens=3):
        self.data = data
        self.coco_path = coco_path
        self.tokenizer = tokenizer
        self.transform = transform
        self.min_label_tokens = min_label_tokens
        # <image> 토큰이 없으면 추가
        if "<image>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_file = sample["image"]
        image_path = f"{self.coco_path}/{image_file}"
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"이미지 로딩 실패: {image_path}, 에러: {e}")
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)
        # 텍스트(첫 human 질문, 첫 gpt 답변)
        question, answer = None, None
        for conv in sample["conversations"]:
            if conv["from"] == "human" and not question:
                question = conv["value"]
            if conv["from"] == "gpt" and not answer:
                answer = conv["value"]
        # 질문/답변이 너무 짧거나 비어 있으면 None 반환(스킵)
        if not question or not answer or len(answer.strip()) < 3:
            return None

        # 프롬프트에 <image> 토큰 자동 삽입
        if "<image>" in question:
            prompt = question
        else:
            prompt = f"<image> {question}"
        max_len = 32
        text_inputs = self.tokenizer(
            prompt,
            padding='max_length', truncation=True, max_length=max_len, return_tensors="pt"
        )
        label_inputs = self.tokenizer(
            answer,
            padding='max_length', truncation=True, max_length=max_len, return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        labels = label_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        # <image> 토큰 위치 찾기 (여러 개면 첫 번째만)
        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        image_token_pos_tensor = (input_ids == image_token_id).nonzero(as_tuple=True)
        positions = image_token_pos_tensor[0]
        if len(positions) == 0:
            return None
        image_token_pos = positions[0].item()

        # labels에서 pad_token을 -100으로 변환
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 라벨(정답)에서 유효 토큰(패딩/무시 제외)이 일정 개수 미만이면 스킵
        valid_label_count = (labels != -100).sum().item()
        if valid_label_count < self.min_label_tokens:
            return None

        # shape 검증
        assert input_ids.shape[0] == labels.shape[0] == attention_mask.shape[0], \
            f"input_ids: {input_ids.shape}, labels: {labels.shape}, attention_mask: {attention_mask.shape}"

        # 샘플 출력 (처음 5개만)
        if idx < 5:
            print(f"\n[샘플 {idx}]")
            print("질문:", question)
            print("프롬프트:", prompt)
            print("답변:", answer)
            print("input_ids[:10]:", input_ids[:10].tolist())
            print("labels[:10]:", labels[:10].tolist())
            print("attention_mask[:10]:", attention_mask[:10].tolist())
            print("<image> 토큰 위치:", image_token_pos)
            print("유효 라벨 토큰 수:", valid_label_count)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_token_pos": torch.tensor(image_token_pos, dtype=torch.long),
        }
