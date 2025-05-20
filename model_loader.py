from langchain_community.llms import HuggingFacePipeline
from transformers import (
   AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch
import os

def load_generation_model():
    
    return HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        device=0 if torch.cuda.is_available() else -1,
        pipeline_kwargs={
            "max_new_tokens": 512,
            "temperature": 0.7
        }
    )

def load_reward_model():
    
    model_path = "reward_model" if os.path.exists("reward_model") else "distilbert-base-uncased"
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Then load model
    model =  AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1
    )
    
    return model, tokenizer  