import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "policy_bot")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.6
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
SIMILARITY_THRESHOLD = 0.3
CORPUS_DIR = "corpus"

# Qdrant mode: "cloud" or "memory"
# Set to "memory" to use in-memory Qdrant (no cloud account needed)
# Set to "cloud" to use Qdrant Cloud (requires QDRANT_URL and QDRANT_API_KEY)
QDRANT_MODE = os.getenv("QDRANT_MODE", "memory")

# Qdrant persistence path for in-memory mode (saves to disk so you don't re-ingest every time)
QDRANT_PERSIST_PATH = os.getenv("QDRANT_PERSIST_PATH", "./qdrant_data")
