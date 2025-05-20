from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import sqlite3
from reward_model import train_reward_model
df = train_reward_model()
# Load the LLM 
model = AutoModelForCausalLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained("reward_model")

# PPO Config (we used PPO because of the small amount of data )
ppo_config = PPOConfig(
    batch_size=32,
    learning_rate=1e-5,
    steps=1000,
)

ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    tokenizer=tokenizer,
)

feedback_dataset = [
        {
            "query": row["query"],
            "response": row["response"],
            "rating": row["rating"]
        }
        for _, row in df.iterrows()
    ]
# Fine-tuning loop 
for batch in feedback_dataset:
    queries, responses, rewards = batch
    
    # Tokenize inputs
    inputs = tokenizer(queries, return_tensors="pt", padding=True)
    
    # Generate responses
    outputs = model.generate(**inputs)
    
    # Calculate rewards
    reward_inputs = tokenizer(
        [f"Query: {q} Response: {r}" for q, r in zip(queries, responses)],
        return_tensors="pt",
        padding=True
    )
    rewards = reward_model(**reward_inputs).logits
    
    # PPO update
    ppo_trainer.step(
        inputs,
        outputs,
        rewards
    )

# Save fine-tuned model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")