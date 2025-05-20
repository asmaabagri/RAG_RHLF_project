import os
os.environ["USER_AGENT"] = "MyRAGApp/1.0"  
import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import WebBaseLoader
import sqlite3
from datetime import datetime
from model_loader import load_generation_model, load_reward_model
from transformers import AutoTokenizer  
from langchain_community.document_loaders import PyPDFLoader
import torch
import re
# Initialize memory to persist across queries
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

CHROMA_PATH = "chroma"
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

PROMPT_TEMPLATE = """
Answer the question based on your knowledge, the following context and past conversation:

Context: {context}

Previous Conversation: {history}

---

Answer the question based on the above context: {question}
"""
# remove any unpaired surrogates (U+D800–U+DFFF)
_surrogate_pattern = re.compile(r'[\ud800-\udfff]')

def clean_text(text: str) -> str:
    return _surrogate_pattern.sub('', text)
#add also website data
URL = ["https://www.ibm.com/think/topics/machine-learning"]
#load the data

data = WebBaseLoader(URL)
#extract the content
web_content = data.load()

for doc in web_content:
    doc.page_content = clean_text(doc.page_content)
db.add_documents(web_content)
generator = load_generation_model()
reward_model, reward_tokenizer = load_reward_model()

# Initialize feedback database
def init_feedback_db():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            rating INTEGER,  
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Log feedback after each response
def log_feedback(query: str, response: str, rating: int):
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (query, response, rating)
        VALUES (?, ?, ?)
    """, (query, response, rating))
    conn.commit()
    conn.close()

def query_rag(query_text: str):
    # Load conversation history
    history = memory.load_memory_variables({})["history"]
    # Add the content of the URL to the database
    if len(db.get()) == 0:
        db.add_documents(web_content)
   
    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5) # First 5 results
    docs_contents = []
    for doc, _score in results:
        snippet = doc.page_content[:1000] 
        docs_contents.append(snippet)
    # Define the context
    context_text = "\n\n---\n\n".join(docs_contents)
    # Born the size of the history
    if isinstance(history, list) and len(history) > 3:
        history = history[-3:]
    # The LLM template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, history=history, question=query_text)
    # Generate the answer
    generator = load_generation_model()
    reward_model, reward_tokenizer = load_reward_model()
        
    response_text = generator.invoke(prompt)
    
    # Score response
    inputs = reward_tokenizer(
        f"Query: {query_text}\nResponse: {response_text}"[:500],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        score = reward_model(**inputs).logits.item()
    
    print(f"Response: {response_text}")
    print(f"Quality score: {score:.2f}/5.0")
    # Save in the memory
    memory.save_context({"input": query_text}, {"output": response_text})

    formatted_response = f"Response: {response_text}"
    print(formatted_response)
    # Save the feedback of the user
  
    try:
        rating = int(input("\nRate the response from 1 (bad) to 5 (excellent): "))
        log_feedback(query_text, response_text, rating)
        print("Feedback saved.")
    except Exception as e:
        print("Feedback not saved (invalid input or skipped).", str(e))
    return response_text

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

if __name__ == "__main__":
    main()

