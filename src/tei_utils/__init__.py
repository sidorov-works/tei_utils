# src/tei_utils/__init__.py

from .client import EncoderClient

# Это то, что будет доступно при "from tei_client import *"
__all__ = [
    "EncoderClient"
]