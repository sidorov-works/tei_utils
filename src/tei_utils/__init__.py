# src/tei_utils/__init__.py

from .encoder_client import EncoderClient
from .prompt_types import PromptType
from .classifier_client import ClassifierClient

# Это то, что будет доступно при "from tei_client import *"
__all__ = [
    "EncoderClient",
    "PromptType",
    "ClassifierClient"
]