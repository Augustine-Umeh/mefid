"""Service clients for API Gateway"""

from .media_processor_client import MediaProcessorClient
from .embedder_client import EmbedderClient
from .indexer_client import IndexerClient

__all__ = [
    "MediaProcessorClient",
    "EmbedderClient",
    "IndexerClient",
]