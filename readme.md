EduRAG: Intelligent Tutor Using RAG and LangChain

A sophisticated AI-powered educational tutoring system that leverages Retrieval-Augmented Generation (RAG), OpenAI's LLM APIs, and PostgreSQL with pgvector for semantic search to deliver intelligent, context-aware educational responses.

🚀 Features

Core Functionality

    Content Upload & Management: Upload educational content with metadata (topic, grade, difficulty)
    Semantic Search: Vector-based similarity search using OpenAI embeddings with pgvector
    RAG Pipeline: Retrieval-Augmented Generation for accurate, context-aware responses
    Multiple Personas: Configurable tutor personalities (friendly, strict, humorous, etc.)
    Natural Language SQL: Convert natural language queries to SQL for database insights
    Interactive Playground: Web-based interface for testing the AI tutor

Technical Features

    PostgreSQL with pgvector: Optimized vector storage and similarity search
    OpenAI Integration: GPT models for answer generation and embedding creation
    Fallback Mechanisms: Local embedding models and OpenAI fallback for reliability
    Production Ready: Nginx + Gunicorn deployment configuration
    Monitoring: Comprehensive logging and health checks

📋 Requirements

System Requirements

    Python 3.12+
    PostgreSQL 16+ with pgvector extension
    Nginx
    Git


Python Dependencies

See requirements/development.txt for full list of dependencies including:



🛠️ Installation & Setup
1. Clone the Repository

git clone git@github.com:anish-gc/edurag-intelligent-teacher-using-rag-and-langchain.git
cd edurag-intelligent-teacher-using-rag-and-langchain


2. Set up Python Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
