import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import clip

class SimpleMultimodalModel(nn.Module):
    def __init__(self, text_model_name="EleutherAI/polyglot-ko-1.3b"):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(text_model_name)
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu')
        for p in self.clip_model.parameters():
            p.requires_grad = False
        hidden_size = self.llm.config.hidden_size
        self.img_proj = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.llm.resize_token_embeddings(self.llm.config.vocab_size + 1)

    def forward(self, input_ids, attention_mask, image, labels=None, image_token_pos=None):
        # 반드시 image_token_pos 인자를 포함!
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image).float()
        image_emb = self.img_proj(image_features)
        text_emb = self.llm.get_input_embeddings()(input_ids)
        multimodal_emb = text_emb.clone()
        for b in range(text_emb.shape[0]):
            multimodal_emb[b, image_token_pos[b]] = image_emb[b]
        outputs = self.llm(inputs_embeds=multimodal_emb, attention_mask=attention_mask, labels=labels)
        return outputs

def load_model_and_tokenizer(model_name="EleutherAI/polyglot-ko-1.3b"):
    model = SimpleMultimodalModel(text_model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
        model.llm.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
