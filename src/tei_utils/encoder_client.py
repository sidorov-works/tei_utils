# src/tei_utils/encoder_client.py

"""
Клиент для работы с TEI-совместимыми сервисами эмбеддингов.
"""

from .tei_models import (
    EmbedRequest,
    embed_response_adaptor,
    TokenizeRequest,
    TokenInfo,
    tokenize_response_adaptor
)
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from pydantic import Field
from .tei_ep_names import EPNames
from .prompt_types import PromptType
from .base_client import BaseClient

import logging
logger = logging.getLogger(__name__)


# Структура InfoResponse содержит не все данные, 
# необходимые для работы клиента. В частности TEI не сообщают 
# в ответе /info важнейший параметр - размерность вектора. 
# Поэтому создадим дополнительную pydantic модель EncoderInfo, 
# в которой данный параметр (dimension) будет присутствовать. 
# Также дополним метаданные именами промптов для типов PromptType
class EncoderExtraInfo(BaseModel):
    """
    Полные метаданные по энкодеру.
    По сравнению со стандартными полями структуры InfoResponse, которую возвращает TEI в /info, содержит:
    - **dimension**: размерность вектора
    - **prompt_names**: отображение типов промптов клиента на соответствующие имена промптов (prompt_name) модели
    """
    dimension: Optional[int] = None
    prompt_names: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {PromptType.QUERY: None, PromptType.DOCUMENT: None}
    )

class EncoderClient(BaseClient):
    """
    Клиент для работы с TEI-совместимыми сервисами эмбеддингов.
    
    При инициализации получает из конфига словарь {имя_энкодера: URL}.
    Для каждого энкодера лениво запрашивает /info при первом использовании.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder_extra_info: Dict[str, Optional[EncoderExtraInfo]] = dict()
    
    # ----------------------------------------------------------------------
    # Приватные методы: получение информации об энкодерах
    # ----------------------------------------------------------------------
    
    async def _get_encoder_extra_info(self, server_name: str) -> Optional[EncoderExtraInfo]:
        """
        Получает дополнительную информацию об энкодере, отсутствующую в InfoResponse.
        
        Извлекает:
        - размерность вектора (через тестовый запрос /embed)
        - имена промптов для query/document (из ответа /info)
        
        Args:
            server_name: Имя энкодера (ключ из словаря servers)
            
        Returns:
            EncoderExtraInfo или None при ошибке
        """
        logger.debug("in _get_encoder_extra_info()")
        # Быстрый возврат, если уже есть
        if encoder_extra := self._encoder_extra_info.get(server_name):
            logger.debug(f"encoder_extra: {encoder_extra}")
            return encoder_extra
        logger.debug("not existing...")
        
        if server_name not in self._servers:
            logger.error(f"Encoder '{server_name}' not found")
            return None
        logger.debug(f"{server_name} in _servers")
        
        async with self._init_locks[server_name]:
            # Double-check после захвата блокировки
            if encoder_extra := self._encoder_extra_info.get(server_name):
                return encoder_extra
            
            # Получаем базовую информацию через родительский метод
            server_info = await self._get_server_info(server_name)
            if not server_info:
                logger.warning(f"Failed to get base info for encoder '{server_name}'")
                return None
            logger.debug(f"server_info: {server_info}")
            
            # Создаем объект для хранения дополнительной информации
            encoder_extra = EncoderExtraInfo()
            
            # 1. Извлекаем имена промптов из ответа /info
            if server_info.prompts:
                for prompt_info in server_info.prompts:
                    if not prompt_info.text:
                        continue
                        
                    name_lower = prompt_info.name.lower()
                    if "query" in name_lower:
                        encoder_extra.prompt_names[PromptType.QUERY] = prompt_info.name
                    elif "document" in name_lower:
                        encoder_extra.prompt_names[PromptType.DOCUMENT] = prompt_info.name

            logger.debug(f"encoder_extra: {encoder_extra}")
            
            # 2. Определяем размерность вектора через тестовый запрос /embed
            client = self._get_http_client(server_name)
            if client:
                try:
                    embed_url = f"{self._servers[server_name]}{EPNames.EMBED}"
                    logger.debug(f"Fetching {server_name} vector dimension at {embed_url}")
                    
                    test_request = EmbedRequest(inputs="i")
                    response = await client.post_with_retry(
                        url=embed_url,
                        json=test_request.model_dump(exclude_none=True),
                        headers={"Content-Type": "application/json"},
                        success_statuses={200}
                    )
                    
                    embeddings = embed_response_adaptor.validate_json(response.text)
                    dimension = len(embeddings[0])
                    encoder_extra.dimension = dimension
                    
                    logger.info(f"✅ Encoder '{server_name}' dimension: {dimension}")
                    
                except Exception as e:
                    logger.warning(f"Failed to get vector dimension for '{server_name}': {e}")
            
            # Сохраняем в отдельный атрибут
            self._encoder_extra_info[server_name] = encoder_extra
            
            logger.info(f"✅ Encoder '{server_name}' extra info: prompts={encoder_extra.prompt_names}, dim={encoder_extra.dimension}")
            
            return encoder_extra

    
    # ----------------------------------------------------------------------
    # Приватные методы: выполнение запросов к TEI
    # ----------------------------------------------------------------------
    
    async def _request_embed(
        self,
        server_name: str,
        inputs: Union[str, List[str]],
        prompt_type: Optional[str] = None,
        prompt_name: Optional[str] = None,
        normalize: bool = True
    ) -> Optional[List[List[float]]]:
        """
        Выполняет запрос к TEI /embed endpoint

        Автоматически делит входной батчи на части в соответствии 
        с ограничением энкодера (у TEI есть max_client_batch_size)

        Args:
            prompt_type: тип промпта "query" или "document". Это не название промпта в энкодере! 
                При переданном prompt_type клиент автоматически выберет подходящее имя промпта. 
                Если prompt_type явно указан, то значение аргумента ptompt_name игнорируется

            prompt_name: имя промпта эмбеддинговой модели. Игнорируется, если явно задан prompt_type
        """
        client = self._get_http_client(server_name) 
        if not client:
            return None
        
        base_info = await self._get_server_info(server_name)
        if not base_info:
            return None
        
        
        if (
            prompt_type and
            (extra_info := await self._get_encoder_extra_info(server_name))
        ):
            prompt_name = extra_info.prompt_names.get(prompt_type)
        
        # Будем учитывать ограничение конкретного TEI на размер батча
        max_batch_size = base_info.max_client_batch_size
        
        try:
            url = f"{self._servers[server_name]}{EPNames.EMBED}"

            # Если в на входе единственный текст, 
            # для единообразия сделаем из него батч длиной 1
            if isinstance(inputs, str):
                inputs = [inputs]

            embeddings_batch: List[List[float]] = []
            # Делим на батчи в соответствии с максимально допустимой длиной батча энкодера
            for i in range(0, len(inputs), max_batch_size):
                # используем срез, не будет ли ошибки с последним "неполным" куском??
                inputs_part = inputs[i: i + max_batch_size]
                # Создаём типизированный запрос
                request = EmbedRequest(
                    inputs=inputs_part,
                    prompt_name=prompt_name,
                    normalize=normalize,
                    truncate=True
                )
                
                embed_response = await client.post_with_retry(
                    url=url,
                    json=request.model_dump(exclude_none=True),
                    headers={"Content-Type": "application/json"}
                )
                
                # Парсим (валидируем) ответ через адаптер
                embeddings = embed_response_adaptor.validate_json(embed_response.text)

                # Убедимся, что длина ответа совпадает с длиной переданных текстов
                if len(embeddings) != len(inputs_part):
                    logger.warning(
                        f"Unexpected {EPNames.EMBED} response length: "
                        f"got {len(embeddings)} while expected {len(inputs_part)}"
                    )
                    return None
                
                embeddings_batch.extend(embeddings)
            
            return embeddings_batch
                
        except Exception as e:
            logger.error(f"TEI embed request to {server_name} failed: {e}")
            return None
        
    
    async def _request_tokenize(
        self,
        server_name: str,
        inputs: Union[str, List[str]]
    ) -> Optional[List[List[TokenInfo]]]:
        """
        Выполняет запрос к TEI /tokenize endpoint.

        Автоматически делит входной батчи на части в соответствии 
        с ограничением энкодера (у TEI есть max_client_batch_size)
        """

        client = self._get_http_client(server_name)
        if not client:
            return None
        
        base_info = await self._get_server_info(server_name)
        if not base_info:
            return None
        
        # Будем учитывать ограничение конкретного TEI на размер батча
        max_batch_size = base_info.max_client_batch_size
        
        try:
            url = f"{self._servers[server_name]}{EPNames.TOKENIZE}"

            # Если в на входе единственный текст, 
            # для единообразия сделаем из него батч длиной 1
            if isinstance(inputs, str):
                inputs = [inputs]


            tokens_batch:List[List[TokenInfo]] = []
            # Делим на батчи в соответствии с максимально допустимой длиной батча энкодера
            for i in range(0, len(inputs), max_batch_size):
                # используем срез, не будет ли ошибки с последним "неполным" куском??
                inputs_part = inputs[i: i + max_batch_size]
                # Создаём типизированный запрос
                request = TokenizeRequest(
                    inputs=inputs_part,
                    add_special_tokens=True,
                    truncate=True
                )
                
                tokenize_response = await client.post_with_retry(
                    url=url,
                    json=request.model_dump(exclude_none=True),
                    headers={"Content-Type": "application/json"}
                )
                
                # Парсим ответ через pydantic адаптер
                tokens: List[List[TokenInfo]] = tokenize_response_adaptor.validate_json(
                    tokenize_response.text
                )

                # Убедимся, что длина ответа совпадает с длиной переданных текстов
                if len(tokens) != len(inputs_part):
                    logger.warning(
                        f"Unexpected {EPNames.TOKENIZE} response length: "
                        f"got {len(tokens)} while expected {len(inputs_part)}"
                    )
                    return None
                
                tokens_batch.extend(tokens)
            
            return tokens_batch
                
        except Exception as e:
            logger.error(f"TEI tokenize request to {server_name} failed: {e}")
            return None
        
    
    # ----------------------------------------------------------------------
    # Публичные методы
    # ----------------------------------------------------------------------
    
    async def encode_text(
        self,
        text: str,
        prompt_type: Optional[str] = None,
        prompt_name: Optional[str] = None,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[List[float]]]:
        """
        Кодирует один текст в вектор.

        Args:
            prompt_type: тип промпта "query" или "document". Это не название промпта в энкодере! 
                При переданном prompt_type клиент автоматически выберет подходящее имя промпта. 
                Если prompt_type явно указан, то значение аргумента ptompt_name игнорируется

            prompt_name: имя промпта эмбеддинговой модели. Игнорируется, если явно задан prompt_type
        
        """
        if use_encoders is None:
            use_encoders = list(self._servers.keys())
        
        results: Dict = await self._request_multiple(
            server_names=use_encoders,
            request_func=self._request_embed,
            inputs=text,
            prompt_type=prompt_type,
            prompt_name=prompt_name,
            normalize=True
        )
        # _request_embed(), как и сам TEI, даже для одиночного текста 
        # возвращает массив массивов. 
        # Но наш публичный метод encode_text() 
        # для одного текста должен возвращать один вектор (для каждого энкодера)
        return {
            encoder: emb[0] if (emb and isinstance(emb, list) and len(emb)) else None 
            for encoder, emb in results.items()
        }
    
    async def encode_batch(
        self,
        texts: List[str],
        prompt_type: Optional[str] = None,
        prompt_name: Optional[str] = None,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[List[List[float]]]]:
        """
        Пакетное кодирование текстов.
        
        Args:
            prompt_type: тип промпта "query" или "document". Это не название промпта в энкодере! 
                При переданном prompt_type клиент автоматически выберет подходящее имя промпта. 
                Если prompt_type явно указан, то значение аргумента ptompt_name игнорируется

            prompt_name: имя промпта эмбеддинговой модели. Игнорируется, если явно задан prompt_type
        """

        if not texts:
            return {encoder: None for encoder in (use_encoders or self._servers)}

        if use_encoders is None:
            use_encoders = list(self._servers.keys())
        
        results = await self._request_multiple(
            server_names=use_encoders,
            request_func=self._request_embed,
            inputs=texts,
            prompt_type=prompt_type,
            prompt_name=prompt_name,
            normalize=True
        )
        
        return results
    
    async def count_tokens(
        self,
        text: str,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[int]]:
        """Подсчитывает количество токенов в тексте."""
        if use_encoders is None:
            use_encoders = list(self._servers.keys())
        
        tokenize_results: Dict[str, Optional[List[List[TokenInfo]]]] = await self._request_multiple(
            server_names=use_encoders,
            request_func=self._request_tokenize,
            inputs=text
        )
        # TEI не имеет отдельного эндпойнта для простого подсчета токенов, 
        # вместо этого он возвращает полную информацию по каждому токену.
        # Поэтому и _request_tokenize() также возвращает детальную информацию 
        # по токенам List[List[TokenInfo]].
        # В данном методе нас интересует только длина текста в токенах,
        # аккуратно извлечем ее из результатов токенизации.
        results = dict()
        for encoder, tokens_batch in tokenize_results.items():
            # Эта функция - для одиночного текста, но _request_tokenize в любом случае 
            # возвращает массив массивов, как будто обрабатывался батч длиной 1. 
            # Поэтому специально называем переменную - tokens_batch 
            # и берем из нее единственный элемент (с индексом 0)
            if tokens_batch and isinstance(tokens_batch, list) and len(tokens_batch):
                tokens_list = tokens_batch[0]
                if tokens_list and isinstance(tokens_list, list):
                    results[encoder] = len(tokens_list)
                    continue
            results[encoder] = None
        
        return results
    
    async def count_tokens_batch(
        self,
        texts: List[str],
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[List[int]]]:
        """
        Подсчитывает количество токенов для нескольких текстов.
        
        Возвращает словарь, в котором ключи - имена энкодеров, 
        а значения - списки, содержащие количества токенов в текстах входного батча
        """
        if not texts:
            return {encoder: None for encoder in (use_encoders or self._servers)}
        
        if use_encoders is None:
            use_encoders = list(self._servers.keys())
        
        tokenize_results: Dict[str, Optional[List[List[TokenInfo]]]] = await self._request_multiple(
            server_names=use_encoders,
            request_func=self._request_tokenize,
            inputs=texts
        )

        # TEI не имеет отдельного эндпойнта для простого подсчета токенов, 
        # вместо этого он возвращает полную информацию по каждому токену.
        # Поэтому и _request_tokenize() также возвращает детальную информацию 
        # по токенам List[List[TokenInfo]].
        # В данном методе нас интересует только длина текста в токенах,
        # аккуратно извлечем ее из результатов токенизации.
        
        results = dict()
        for encoder, tokens_batch in tokenize_results.items():
            if tokens_batch and isinstance(tokens_batch, list) and len(tokens_batch):
                results[encoder] = [len(tokens) for tokens in tokens_batch if isinstance(tokens, list)]
            else:
                results[encoder] = None
        
        return results
    
    # ----------------------------------------------------------------------
    # Методы для получения информации 
    # ----------------------------------------------------------------------
    
    async def get_vector_size(self, server_name: str) -> Optional[int]:
        """Возвращает размерность вектора."""
        logger.debug("Before _get_encoder_extra_info()")
        extra_info = await self._get_encoder_extra_info(server_name)
        return extra_info.dimension if extra_info else None
    
    async def get_vector_sizes(
        self,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[int]]:
        """
        Возвращает размерности векторов для нескольких энкодеров.
        """
        if use_encoders is None:
            use_encoders = list(self._servers.keys())
        
        # Создаем задачи для параллельного выполнения
        tasks = {
            name: self._get_encoder_extra_info(name)
            for name in use_encoders
        }
        
        # Запускаем все задачи параллельно
        results = {}
        for name, task in tasks.items():
            try:
                extra_info = await task
                results[name] = extra_info.dimension if extra_info else None
            except Exception as e:
                logger.error(f"Failed to get vector size for {name}: {e}")
                results[name] = None
        
        return results