"""Centralized Qdrant client factory. Supports both cloud and local/in-memory modes."""
import atexit
from qdrant_client import QdrantClient
from config import QDRANT_MODE, QDRANT_URL, QDRANT_API_KEY, QDRANT_PERSIST_PATH

# Monkey-patch __del__ on QdrantClient to prevent the known Windows Python shutdown
# bug where sys.meta_path is None when garbage collection runs deallocators.
_orig_del = getattr(QdrantClient, "__del__", None)


def _safe_del(self):
    try:
        if hasattr(self, "close"):
            self.close()
    except Exception:
        pass


QdrantClient.__del__ = _safe_del

_client_instance = None


def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client singleton based on configured mode."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    if QDRANT_MODE == "cloud" and QDRANT_URL and QDRANT_API_KEY:
        _client_instance = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60,
            check_compatibility=False
        )
    else:
        _client_instance = QdrantClient(path=QDRANT_PERSIST_PATH)
        
    atexit.register(close_qdrant_client)
    return _client_instance


def close_qdrant_client():
    """Explicit clean shutdown hook."""
    global _client_instance
    if _client_instance is not None:
        try:
            _client_instance.close()
        except Exception:
            pass
        _client_instance = None
