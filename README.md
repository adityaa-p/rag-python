### RAG application that lets you query the contents of a PDF (Python Handbook) using natural language.

It works by:
- Loading and chunking the PDF into manageable pieces
- Creating vector embeddings of the chunks
- Storing and retrieving chunks using Qdrant vector db (running locally)
- Passing the most relevant context to GPT-4 to generate accurate answers
