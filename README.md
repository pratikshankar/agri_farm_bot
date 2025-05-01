# 🌾 AgriBot — Your Smart Agriculture & Dairy Assistant 🤖🐄

AgriBot is an intelligent chatbot built to answer **farm-specific** as well as **general agriculture and dairy-related** queries. Whether you're a dairy farmer needing insights on milk yield, cattle health, or a cultivator looking for crop disease help — AgriBot is trained to assist with ease.

---
![alt text](<Screenshot 2025-05-01 at 5.14.47 PM.png>)

## 🧠 Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** – API backend for managing routes and logic
- **[LangChain](https://www.langchain.com/)** – Orchestration framework for LLMs
- **[Hugging Face Transformers](https://huggingface.co/)** – Multilingual Embedding model
- **Google Gemini API (LLM)** – Handles deep language reasoning and responses
- **[Pinecone](https://www.pinecone.io/)** – Vector database for document indexing and similarity search
- **HTML + Bootstrap** – Frontend chat interface

---

## 🐮 Features

- ✅ **Chat with farm-specific data** (PDFs & CSVs supported)
- ✅ Multilingual support (Marathi, Hindi, Kannada, and more)
- ✅ Dairy-specific analytics (e.g., milk yield per cow, estrus cycle tracking)
- ✅ General agriculture & crop disease queries
- ✅ Seamless integration with local files via embeddings
- ✅ Natural, LLM-powered summarizations

---

## 🚀 Setup Guide

```bash
# 1. Clone the repository
git clone https://github.com/pratikshankar/agri_farm_bot.git
cd agri_farm_bot

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set your environment variables
export GOOGLE_API_KEY="your_google_gemini_key"
export PINECONE_API_KEY="your_pinecone_key"
export PINECONE_ENV="your_pinecone_env"
export HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"

# 4. Run the backend FastAPI server
uvicorn main:app --reload
