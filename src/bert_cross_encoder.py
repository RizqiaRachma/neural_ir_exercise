"""
BERT Cross-Encoder for Neural Re-Ranking
Dosen: Zico Pratama Putra
Kelompok: Rizqia - Tungkot - Intan
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import os

class CrossEncoderDataset(Dataset):
    def __init__(self, queries, passages, labels, tokenizer, max_length=512):
        self.queries = queries
        self.passages = passages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.queries[idx],
            self.passages[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BERTCrossEncoder:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        print(f"✅ Model {model_name} loaded on {self.device}")

    def predict(self, query: str, passage: str) -> float:
        """Inference satu pasangan query-passage"""
        features = self.tokenizer(query, passage, truncation=True, padding=True, return_tensors="pt")
        features = {k: v.to(self.device) for k, v in features.items()}
        
        with torch.no_grad():
            outputs = self.model(**features)
            # [PROTEKSI FIX INDEX ERROR DIMENSI]
            if outputs.logits.shape[1] == 1:
                score = torch.sigmoid(outputs.logits).item()
            else:
                score = outputs.logits.softmax(dim=1)[:, 1].item()
        return score

    def re_rank(self, query: str, passages: list, batch_size=32):
        """Re-rank list of passages untuk satu query"""
        scores = []
        for i in tqdm(range(0, len(passages), batch_size), desc="Re-ranking"):
            batch_queries = [query] * min(batch_size, len(passages) - i)
            batch_passages = passages[i:i + batch_size]
            
            batch_scores = [self.predict(q, p) for q, p in zip(batch_queries, batch_passages)]
            scores.extend(batch_scores)
        
        # Return ranked indices + scores
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_indices, scores

    def train(self, train_df: pd.DataFrame, val_df=None, epochs=1, batch_size=8, lr=2e-5):
        """[LENGKAP] Implementasi Full Training Loop berbasis Classification Loss"""
        train_dataset = CrossEncoderDataset(train_df['query'], train_df['passage'], train_df['label'], self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss() if self.model.config.num_labels > 1 else nn.BCEWithLogitsLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels) if self.model.config.num_labels > 1 else criterion(outputs.logits.squeeze(-1), labels.float())
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} Selesai. Average Loss: {total_loss / len(train_loader):.4f}")

# ====================== CONTOH PENGGUNAAN ======================
if __name__ == "__main__":
    reranker = BERTCrossEncoder()
    
    score = reranker.predict(
        query="How to make a good cappuccino?",
        passage="The three steps to make a perfect cappuccino are..."
    )
    print(f"Relevance score: {score:.4f}")