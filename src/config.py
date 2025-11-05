# config.py

# Model Provider Selection
USE_OPENAI = False  # Set to True to use OpenAI, False to use Ollama

if USE_OPENAI:
    # OpenAI/SiliconFlow Configuration
    base_url = "https://api.siliconflow.cn/v1"  # SiliconFlow API endpoint
    OPENAI_API_KEY = "your_api_key"  # Set your actual API key here
    OPENAI_MODEL = "Qwen/Qwen2.5-72B-Instruct"
    OPENAI_EMBEDDING_MODEL = "BAAI/bge-m3"
    OPENAI_SIMILARITY_MODEL = "Qwen/Qwen2.5-14B-Instruct"
    
    # Not used when USE_OPENAI=True
    OLLAMA_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "qwen2.5:72b"
    EMBEDDING_MODEL = "bge-m3:latest"
    SIMILARITY_MODEL = "qwen2:7b"
else:
    # Ollama Configuration (Local in Colab)
    OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama running locally in Colab
    DEFAULT_MODEL = "qwen2.5:72b"
    EMBEDDING_MODEL = "bge-m3:latest"
    SIMILARITY_MODEL = "qwen2:7b"
    
    # Not used when USE_OPENAI=False
    base_url = None
    OPENAI_API_KEY = None
    OPENAI_MODEL = None
    OPENAI_EMBEDDING_MODEL = None
    OPENAI_SIMILARITY_MODEL = None