from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

def get_bert_embeddings(
    texts, 
    model_name='distilbert-base-uncased', 
    max_length=256, 
    batch_size=16, 
    device='cpu'
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = []
    # tqdm은 오직 BERT 임베딩 추출에서만 사용!
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT 임베딩 추출"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length, 
            padding='max_length'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_emb = outputs.last_hidden_state[:,0,:].cpu().numpy()
            embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings)
    return embeddings
