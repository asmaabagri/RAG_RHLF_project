from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import torch
from sklearn.model_selection import train_test_split
import sqlite3
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_reward_model():
    # Load feedback data
    conn = sqlite3.connect("feedback.db")
    df = pd.read_sql_query("SELECT query, response, rating FROM feedback", conn)
    conn.close()
    if len(df) < 10:  # Minimum feedback threshold
        raise ValueError(f"Not enough feedback data ({len(df)} samples). Need at least 10 rated examples.")
    
    # Preprocess data 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
    optimizer = AdamW(model.parameters(), lr=5e-5) 
    texts = [f"Query: {row['query']} Response: {row['response']}" for _, row in df.iterrows()]
    ratings = torch.tensor(df["rating"].values / 5.0, dtype=torch.float32)  # Normalized
    
    # Tokenize all texts
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    # Split the data
    input_ids_train, input_ids_val, attention_mask_train, attention_mask_val, y_train, y_val = train_test_split(
        encodings["input_ids"], encodings["attention_mask"], ratings, test_size=0.2, random_state=42
    )
   
    # Create TensorDatasets
    train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train)
    val_dataset = TensorDataset(input_ids_val, attention_mask_val, y_val)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    # Training loop 
    model.train()
    for epoch in range(3):  # 3 epochs
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.unsqueeze(1))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} - Avg Loss: {total_loss/len(train_loader):.4f}")

    # Save model 
    model.save_pretrained("reward_model")
    tokenizer.save_pretrained("reward_model")
    print("Reward model saved to 'reward_model' directory")
    return df 

if __name__ == "__main__":
    train_reward_model()
