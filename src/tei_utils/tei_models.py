# src/tei_utils/tei_models.py

"""
Pydantic модели для TEI-совместимого API
"""

from pydantic import BaseModel, Field, ConfigDict, TypeAdapter
from typing import Union, List, Optional, Literal

class NestedBase(BaseModel):
    model_config = ConfigDict(
        extra='ignore', # незнакомые поля просто игнорируются
        from_attributes=True
    )

# ===========================================================================
# /embed
# ===========================================================================

class EmbedRequest(NestedBase):
    """
    Запрос к /embed endpoint в формате TEI.
    
    Поддерживает как одиночные тексты, так и батчи через поле inputs.

    Наследуем от NestedBase, тем самым допуская, что клиент может передать 
    дополнительные и неизвестные нашему Encoder Service поля, 
    которые будут просто игнорироваться.
    """
    
    inputs: Union[str, List[str]] = Field(
        ..., 
        description="Text or list of texts to embed"
    )

    prompt_name: Optional[str] = Field(
        None, 
        description="Name of prompt template (e.g., 'query', 'document')"
    )
    normalize: Optional[bool] = Field(
        False, 
        description="Whether to normalize embeddings to unit length"
    )
    truncate: Optional[bool] = Field(
        True, 
        description="Whether to truncate inputs to max_input_length"
    )
    truncation_direction: Optional[Literal['left', 'right']] = Field(
        "right", 
        description="Truncate the right or the left side"
    )


# Ответ TEI не содержит именованных полей - всегда возвращается 
# массив массивов (векторов) - List[List[float]], 
# даже если передавался только один текст).
# Валидировать можно с помощью pydantic TypeAdaptor: 
# embeddings: List[List[float]] = embed_response_adaptor.validate_json(response.text)
embed_response_adaptor = TypeAdapter(List[List[float]])


# ===========================================================================
# /tokenize
# ===========================================================================

class TokenizeRequest(NestedBase):
    """
    Запрос к /tokenize endpoint в формате TEI.

    Поддерживает как одиночные тексты, так и батчи через поле inputs.

    Наследуем от NestedBase, тем самым допуская, что клиент может передать 
    дополнительные и неизвестные нашему Encoder Service поля, 
    которые будут просто игнорироваться.
    """

    inputs: Union[str, List[str]] = Field(
        ..., 
        description="Text or list of texts to tokenize"
    )
    
    add_special_tokens: Optional[bool] = Field(
        True, 
        description="Whether to add special tokens to the tokenized output"
    )
    truncate: Optional[bool] = Field(
        True, 
        description="Whether to truncate inputs to max_input_length"
    )

class TokenInfo(NestedBase):
    """
    Информация об одном токене в формате TEI.
    
    Поля start/stop могут отсутствовать (None), если токенизатор
    не поддерживает возврат позиций в исходном тексте.
    """
    id: int = Field(..., description="Token ID in vocabulary")
    text: str = Field(..., description="Token text")
    special: bool = Field(..., description="Whether this is a special token ([CLS], [SEP], etc.)")
    start: Optional[int] = Field(None, description="Start character position in original text")
    stop: Optional[int] = Field(None, description="End character position in original text")

# Результат /tokenize представляет собой массив массивов 
# структур TokenInfo - List[List[TokenInfo]]. 
# При этом сами массивы не являются именованными полями - 
# то есть опять не получается сделать pydantic модель ответа

tokenize_response_adaptor = TypeAdapter(List[List[TokenInfo]])


# ===========================================================================
# /info
# ===========================================================================

# GET запрос к /info не требует параметров

class PromptInfo(BaseModel):
    """Информация о промпте"""
    name: str = Field(..., description="Prompt name to use in prompt_name field")
    text: str = Field(..., description="Text that will be prepended to input")

class InfoResponse(NestedBase):
    """
    Ответ от /info endpoint в формате TEI.
    
    Содержит только те поля, которые реально нужны для работы клиента:
    - model_id: идентификация модели
    - max_input_length: ограничение на длину текста
    - max_client_batch_size: ограничение на размер батча
    - prompts: список промптов, доступных для модели 
    
    Размерность вектора (dimension) не входит в официальную спецификацию TEI.
    Клиент должен определять её отдельно через тестовый запрос к /embed.
    
    Наследуем от NestedBase, допуская что сервер может вернуть
    дополнительные поля - они будут просто игнорироваться.
    """
    model_id: str = Field(
        ..., 
        description="Hugging Face model ID"
    )
    max_input_length: Optional[int] = Field(
        None, 
        description="Maximum input length in tokens"
    )
    max_client_batch_size: int = Field(
        32,
        description="Maximum number of texts allowed in a single /embed request. "
                    "Client must split larger batches." 
    )
    prompts: List[PromptInfo] = Field(None, description="Available prompts for this model")