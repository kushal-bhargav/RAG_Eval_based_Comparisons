# config.py - Colab optimized version
USE_OPENAI = False

if USE_OPENAI:
    base_url = "https://api.siliconflow.cn/v1"
    OPENAI_API_KEY = "your_api_key"
    OPENAI_MODEL = "Qwen/Qwen2.5-72B-Instruct"
    OPENAI_EMBEDDING_MODEL = "BAAI/bge-m3"
    OPENAI_SIMILARITY_MODEL = "Qwen/Qwen2.5-14B-Instruct"
    
    OLLAMA_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "qwen2.5:7b"  # Changed from 72b
    EMBEDDING_MODEL = "bge-m3:latest"
    SIMILARITY_MODEL = "qwen2:7b"
else:
    # Ollama Configuration (optimized for Colab)
    OLLAMA_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "qwen2.5:7b"  # Changed from 72b to 7b
    EMBEDDING_MODEL = "bge-m3:latest"
    SIMILARITY_MODEL = "qwen2:7b"
    
    base_url = None
    OPENAI_API_KEY = None
    OPENAI_MODEL = None
    OPENAI_EMBEDDING_MODEL = None
    OPENAI_SIMILARITY_MODEL = None
