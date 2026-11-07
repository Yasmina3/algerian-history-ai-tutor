# 🇩🇿 Algerian History AI Tutor
### Domain-Specific LLM Fine-Tuning, Retrieval-Augmented Generation (RAG) & Tool-Augmented LLM Architecture

An advanced Natural Language Processing (NLP) project combining **Parameter-Efficient Fine-Tuning (PEFT), semantic retrieval, vector databases, and multi-model orchestration** to develop an Arabic-speaking AI tutor specialized in Algerian history.

The system combines a lightweight, domain-adapted language model with a retrieval pipeline and a larger instruction-following LLM to generate context-aware, educational responses grounded in historical learning materials.

---

## 🎯 Project Overview

The objective is to build an intelligent educational assistant capable of answering historical questions in Arabic, with particular emphasis on the Algerian educational curriculum.

Rather than relying exclusively on a general-purpose language model, the project explores a hybrid architecture combining:

- Domain-specific LLM fine-tuning.
- Retrieval-Augmented Generation (RAG).
- Semantic search over structured historical documents.
- Hierarchical knowledge organization based on educational levels.
- Function calling to integrate specialized models into a larger LLM.
- Multi-model collaboration for improved response generation.

## 🏗️ System Architecture

The solution is organized into three major components.

### 1. Domain-Specific LLM Fine-Tuning

A Meta Llama 3.2 3B Instruct model is adapted to Arabic historical question answering using supervised fine-tuning.

**Implementation:**
- Dataset preparation using approximately 9,722 question-answer pairs.
- Arabic instruction formatting and contextual prompt engineering.
- Parameter-Efficient Fine-Tuning using LoRA.
- 4-bit quantization to reduce GPU memory requirements.
- Supervised Fine-Tuning using Hugging Face TRL.
- Training optimization using Unsloth.
- Model and tokenizer publication through Hugging Face Hub.

**Training configuration:**
- Base model: Llama 3.2 3B Instruct.
- Fine-tuning: LoRA / PEFT.
- LoRA rank: 16.
- Sequence length: 2,048 tokens.
- Quantization: 4-bit.
- Training epochs: 10.

### 2. Retrieval-Augmented Generation (RAG)

A semantic retrieval pipeline provides access to structured historical knowledge and educational materials.

**Data preparation:**
- Processing Arabic historical educational content.
- Structuring documents using titles, historical periods, and content metadata.
- Paragraph aggregation and document segmentation.
- Organizing historical content into educational levels.
- Preparing documents for embedding and indexing.

**Retrieval pipeline:**
1. Preprocess the Arabic user query.
2. Generate semantic embeddings using E5-based embedding models.
3. Search the Pinecone vector database.
4. Retrieve relevant historical passages.
5. Inject retrieved context into the generation pipeline.

**Technologies:**
- Hugging Face Datasets.
- Semantic Router.
- E5 embedding models.
- Pinecone Vector Database.
- Cosine similarity search.

### 3. Multi-Model Integration & Intelligent Chatbot

The final component combines the fine-tuned domain-specific model with a larger language model.

The architecture uses a Llama 70B model hosted through Groq as the main conversational model.

The specialized Llama 3.2 3B model is exposed as a callable function/tool, allowing the larger model to request domain-specific historical knowledge when appropriate.

**Integration workflow:**
1. Receive a historical question.
2. Retrieve relevant historical documents through RAG.
3. Provide the larger LLM with the user query and retrieved context.
4. Allow the orchestrating model to invoke the specialized fine-tuned model through function calling.
5. Integrate the available information.
6. Generate a detailed Arabic response.

This architecture explores the complementary strengths of domain-specific fine-tuning and large-scale instruction-following models.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| NLP & LLMs | Hugging Face Transformers, Meta Llama 3.2 |
| Fine-Tuning | Unsloth, PEFT, LoRA |
| Training | Hugging Face TRL |
| Quantization | BitsAndBytes, 4-bit |
| Embeddings | E5-based Embedding Models |
| Vector Database | Pinecone |
| Retrieval | Semantic Search, RAG |
| LLM Inference | Groq API |
| Dataset Management | Hugging Face Datasets |
| Experiment Tracking | Weights & Biases |
| Environment | Google Colab / GPU |

---

## 📊 Datasets & Models

The project uses two complementary types of data:

**1. Fine-tuning dataset**

Approximately 9,722 Arabic historical question-answer pairs, structured with educational-level and historical-period information.

**2. Retrieval knowledge base**

Structured historical educational materials segmented into semantic chunks, with metadata for educational levels, titles, and historical contexts.

The notebook documents an approximately 200,000-chunk retrieval corpus.

**Models:**

- Fine-tuned model: [History-final-Llama-3.2-3B-Instruct-bnb-4bit](https://huggingface.co/yasmina3/History-final-Llama-3.2-3B-Instruct-bnb-4bit)
- Embedding models: E5-based Arabic-compatible embedding models.
- Orchestrating LLM: Llama 70B through Groq.

---

## 💻 Repository Structure

```text
algerian-history-ai-tutor/
│
├── algerian-history-llm-finetuning-rag.ipynb
│
├── README.md
│
├── requirements.txt
│
└── data/
    └── README.md
```

The notebook contains the experimental implementation of the complete workflow, including model fine-tuning, retrieval infrastructure, and chatbot integration.

Datasets and model artifacts are managed separately through Hugging Face Hub and external storage.

---

## 🚀 Running the Project

The original implementation was developed primarily in Google Colab and requires a GPU-enabled environment for model fine-tuning and inference.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/algerian-history-ai-tutor.git

cd algerian-history-ai-tutor
```

### 2. Install dependencies

The notebook uses the following core libraries:

```bash
pip install unsloth
pip install transformers datasets trl peft
pip install bitsandbytes accelerate
pip install pinecone-client
pip install semantic-router
pip install groq
pip install wandb
```

Some dependencies and API integrations may require version adjustments depending on the execution environment.

### 3. Configure API credentials

Configure the required credentials as environment variables:

```python
import os

os.environ["HF_TOKEN"] = "your_huggingface_token"
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key"
os.environ["WANDB_API_KEY"] = "your_wandb_api_key"
```

Never commit API keys or private credentials to GitHub.

### 4. Execute the notebook

Open the notebook in Google Colab or a compatible GPU-enabled Jupyter environment.

The notebook documents the three main stages:

1. Fine-tuning the domain-specific LLM.
2. Building and testing the RAG pipeline.
3. Integrating the models into the conversational assistant.

**Note:** Full reproduction requires access to the original datasets, compatible model artifacts, API credentials, and the corresponding vector indexes.

---

## 🔬 Technical Contributions

The main technical contributions explored in this project include:

- Adapting a pretrained open-source LLM to a specialized Arabic historical domain.
- Applying LoRA-based fine-tuning and low-bit quantization to optimize model adaptation.
- Building a semantic retrieval pipeline for Arabic educational documents.
- Implementing metadata-aware knowledge organization across educational levels.
- Combining parametric knowledge from a fine-tuned model with externally retrieved information.
- Integrating specialized LLM capabilities into a larger model through function calling.
- Exploring hybrid LLM architectures for domain-specific educational applications.

---

## 👩‍💻 Author

**Yasmina Medjri**

AI & Data Science Engineer | Generative AI | NLP | Machine Learning

GitHub: [@Yasmina3](https://github.com/Yasmina3)

Hugging Face: [@yasmina3](https://huggingface.co/yasmina3)

---

*An exploration of domain-adapted language models, Arabic NLP, semantic retrieval, and hybrid Generative AI architectures.*
