import os
from utils import get_embedding_model, get_vector_store
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

user_query = input("❓ Enter your question: ")
k = 5  # Top K results

embedding_model = get_embedding_model()
collection_name = os.getenv("COLLECTION_NAME")
vector_db_url = os.getenv("VECTOR_DB_URL")

vector_db = get_vector_store(embedding_model, collection_name, vector_db_url)
results = vector_db.similarity_search(user_query, k=k)

context = "\n\n".join([
    f"[Page {r.metadata['page_label']}]: {r.page_content.strip()}"
    for r in results
])

SYSTEM_PROMPT = f"""You are an assistant that answers based only on the provided context from a Python Handbook.

If you cannot answer from the context, respond with "I'm not sure. Please check the handbook."

Context:
{context}
"""

client = OpenAI()

chat_completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
)

print(f"\n🤖 {chat_completion.choices[0].message.content}")