import torch
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from dataset import VisualInstructDataset
from model import load_model_and_tokenizer

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("tabtoyou/KoLLaVA-Instruct-150k")
    train_data = ds["train"].select(range(20000))
    indices = list(range(len(train_data)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_split = train_data.select(train_idx)
    val_split = train_data.select(val_idx)

    COCO_PATH = "/home/hanborim/mbti_project/coco/train2017"
    model, tokenizer = load_model_and_tokenizer()
    model = model.to(device)
    print("실제 모델 클래스:", model.__class__)
    print("forward 인자:", model.forward.__code__.co_varnames)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # min_label_tokens=3로 유효 답변만 학습
    train_dataset = VisualInstructDataset(train_split, COCO_PATH, tokenizer, transform, min_label_tokens=3)
    val_dataset = VisualInstructDataset(val_split, COCO_PATH, tokenizer, transform, min_label_tokens=3)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        list(model.img_proj.parameters()) + list(model.llm.parameters()), lr=1e-5
    )
    num_epochs = 5  # 실제 학습에서는 5 이상 권장

    best_val_loss = float('inf')
    best_path = "my_multimodal_model_best.pt"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch} [Train]") as pbar:
            for i, batch in enumerate(pbar):
                if batch is None:
                    continue
                try:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    image = batch["image"].to(device)
                    labels = batch["labels"].to(device)
                    image_token_pos = batch["image_token_pos"].to(device)

                    optimizer.zero_grad()
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image=image,
                        labels=labels,
                        image_token_pos=image_token_pos
                    )
                    loss = outputs.loss

                    if torch.isnan(loss):
                        tqdm.write(f"[Batch {i}] loss is nan! 해당 배치 스킵")
                        continue  # nan이면 해당 배치만 건너뛰고 계속 진행

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                except Exception as e:
                    tqdm.write(f"에러 발생 (Batch {i}): {e}")

        avg_train_loss = train_loss / len(train_loader)

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                image = batch["image"].to(device)
                labels = batch["labels"].to(device)
                image_token_pos = batch["image_token_pos"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image=image,
                    labels=labels,
                    image_token_pos=image_token_pos
                )
                loss = outputs.loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # === BEST CHECKPOINT 저장 ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Best model updated at epoch {epoch} (val_loss={avg_val_loss:.4f})")

    tokenizer.save_pretrained("my_tokenizer_dir")
    model.llm.save_pretrained("my_multimodal_model_best")
    tokenizer.save_pretrained("my_tokenizer_dir")

    print("가장 성능 좋은 모델만 저장 완료!")

if __name__ == "__main__":
    train()
