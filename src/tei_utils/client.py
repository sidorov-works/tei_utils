# src/tei_utils/client.py

"""
Клиент для работы с TEI-совместимыми сервисами эмбеддингов.
"""

from .tei_models import (
    EmbedRequest,
    embed_response_adaptor,
    TokenizeRequest,
    TokenInfo,
    tokenize_response_adaptor,
    InfoResponse
)
from typing import List, Optional, Dict, Any, Union
import asyncio
from http_utils import RetryableHTTPClient, create_signed_client, AuthType
from pydantic import Field

import logging
logger = logging.getLogger(__name__)

class EPNames:
    """
    Названия используемых эндпойнтов TEI
    """
    EMBED = "/embed"                   
    TOKENIZE = "/tokenize"
    INFO = "/info"
    HEALTH = "/health"

class PromptType:
    """
    Типы промптов для TEI.

    ВАЖНО: тип промптра - это НЕ prompt_name для энкодера, 
    а просто указание на желаемый промпт, 
    имя которого должно быть выяснено отдельно
    """
    QUERY = "query"
    DOCUMENT = "document"


# Структура InfoResponse содержит не все данные, 
# необходимые для работы клиента. В частности TEI не сообщают 
# в ответе /info важнейший параметр - размерность вектора. 
# Поэтому создадим дополнительную pydantic модель EncoderInfo, 
# в которой данный параметр (dimension) будет присутствовать. 
# Также дополним метаданные именами промптов для типов PromptType
class EncoderInfo(InfoResponse):
    """
    Полные метаданные по энкодеру. Включает в себя
    """
    dimension: Optional[int] = None
    prompt_names: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {PromptType.QUERY: None, PromptType.DOCUMENT: None}
    )

class EncoderClient:
    """
    Клиент для работы с TEI-совместимыми сервисами эмбеддингов.
    
    При инициализации получает из конфига словарь {имя_энкодера: URL}.
    Для каждого энкодера лениво запрашивает /info при первом использовании.
    
    """
    
    def __init__(
            self,
            encoders: Dict[str, str],
            secret: str,
            request_timeout: float = 30.0,
            total_timeout: float = 60.0
        ):
        """
        Инициализация клиента.

        Args:
            encoders: словарь, ключи которого являются именами TEI сервисов, 
                а значения - их URL.

            secret: секретный ключ для аутентификации в TEI сервисах

            request_timeout: тайм-аут на выполнение запроса к TEI (без учета ретраев)

            total_timeout: общий тайм-аут на выполнени запроса к TEI с ретраями 
                (включает как отдельные попытки, так и задержки между попытками)
        """

        self._secret = secret

        self._request_timeout = request_timeout
        self._total_timeout = total_timeout

        if encoders and isinstance(encoders, dict):
            self._encoders: Dict[str, str] = encoders.copy()    
        else:
            # Если не передали данные сервисов, то дальше инициализировать нечего
            logger.warning("EncoderClient initialized with no encoders")
            self._encoder_info = dict()
            self._http_clients = dict()
            self._init_locks = dict()
            return
        
        # Сюда попадаем, если какие-то энкодеры переданы

        # Кэш метаданных энкодеров
        self._encoder_info: Dict[str, Optional[EncoderInfo]] = {
            name: None for name in self._encoders
        }
        
        # HTTP клиенты для каждого энкодера
        self._http_clients: Dict[str, Optional[RetryableHTTPClient]] = {
            name: None for name in self._encoders
        }
        
        # Блокировки для безопасной ленивой инициализации
        self._init_locks: Dict[str, asyncio.Lock] = {
            name: asyncio.Lock() for name in self._encoders
        }
        
        logger.debug(f"EncoderClient initialized with encoders: {list(self._encoders.keys())}")
    
    # ----------------------------------------------------------------------
    # Приватные методы: инициализация HTTP клиентов
    # ----------------------------------------------------------------------
    
    def _get_http_client(self, encoder_name: str) -> Optional[RetryableHTTPClient]:
        """
        Ленивое получение HTTP клиента для базовых запросов.

        TEI использует базовую аутентификацию запросов:
        `auth_header_name: str = "Authorization"`, `auth_header_scheme: str = "Bearer"`

        Возвращает None, если имя encoder_name не знакомо клиенту.
        """
        # Возвращаем None, если имя encoder_name не знакомо
        if encoder_name not in self._http_clients:
            logger.error(f"Unknown encoder: {encoder_name}")
            return None
        
        if self._http_clients[encoder_name] is None:
            client = RetryableHTTPClient(
                base_timeout=self._request_timeout,
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                total_timeout=self._request_timeout * 2
            )
            # Оборачиваем http клиента для автоматического подписания запросов
            self._http_clients[encoder_name] = create_signed_client(
                client,
                secret=self._secret,
                service_name="encoder-client",
                auth_type=AuthType.SECRET_HEADER_AUTH
            )
        
        return self._http_clients[encoder_name]
    
    
    # ----------------------------------------------------------------------
    # Приватные методы: получение информации об энкодерах
    # ----------------------------------------------------------------------
    
    async def _get_encoder_info(self, encoder_name: str) -> Optional[EncoderInfo]:
        """
        Получает основную информацию об энкодере `encoder_name` через `"/info"`,
        дополняет ее параметром dimension, который приходится выяснять 
        через отдельный вызов `"/embed"` (оригинальный *TEI* не включает 
        информацию о размерности вектора эмбеддинговой модели в ответ `"/info"`).

        Возвращает `EncoderInfo` - это `InfoResponse` с доп. атрибутом `dimension`.
        """
        if self._encoder_info.get(encoder_name) is not None:
            return self._encoder_info[encoder_name]
        
        if encoder_name not in self._encoders:
            logger.error(f"Encoder '{encoder_name}' not found in config")
            return None
        
        async with self._init_locks[encoder_name]:
            if self._encoder_info[encoder_name] is not None:
                return self._encoder_info[encoder_name]
            
            client = self._get_http_client(encoder_name)
            if not client:
                return None
            
            try:
                info_url = f"{self._encoders[encoder_name]}{EPNames.INFO}"
                logger.debug(f"Fetching info from {encoder_name} at {info_url}")
                
                info_response = await client.get_with_retry(
                    url=info_url,
                    success_statuses={200}
                )
                
                data = info_response.json()
                # Запоминаем основные данные энкодера (пока без dimension)
                encoder_info = EncoderInfo.model_validate(data)

                # Если присутствует информация о промптах эмбеддинговой модели, 
                # найдем имена промптов (prompt_name) для применяемых типов 
                # PromptType.QUERY и PromptType.DOCUMENT
                if encoder_info.prompts:
                    for prompt_info in encoder_info.prompts:
                        if prompt_info.text:
                            if "query" in prompt_info.name.lower():
                                encoder_info.prompt_names[PromptType.QUERY] = prompt_info.name
                            elif "document" in prompt_info.name.lower():
                                encoder_info.prompt_names[PromptType.DOCUMENT] = prompt_info.name

                # Теперь с помощью отдельного запроса /embed выясним размерность вектора
                embed_url = f"{self._encoders[encoder_name]}{EPNames.EMBED}"
                logger.debug(f"Fetching {encoder_name} vector dimension at {embed_url}")
                TEST_TEXT = "i"
                embed_request = EmbedRequest(inputs=TEST_TEXT)
                embed_response = await client.post_with_retry(
                    url=embed_url,
                    json=embed_request.model_dump(exclude_none=True),
                    headers={"Content-Type": "application/json"},
                    success_statuses={200}
                )
                embeddings = embed_response_adaptor.validate_json(embed_response.text)
                dimension = len(embeddings[0])
                encoder_info.dimension = dimension
                # Теперь информация о сервисе (энкодере) - полная: INFO_RESPONSE плюс dimension и prompt_names
                self._encoder_info[encoder_name] = encoder_info
                
                logger.info(f"✅ Encoder '{encoder_name}' initialized: {encoder_info.model_id}")
                return encoder_info
                
            except Exception as e:
                logger.warning(f"Failed to get info from encoder '{encoder_name}': {e}")
                return None
            
    async def _check_encoder_health(self, encoder_name: str) -> bool:
        http_client = self._get_http_client(encoder_name)
        if http_client:
            try:
                health_url = f"{self._encoders[encoder_name]}{EPNames.HEALTH}"
                health_response = await http_client.get_with_retry(health_url)
                return True
            except:
                logger.debug(f"Failed to check encoder {encoder_name} /health")
                return False
        else:
            logger.debug(f"Unknown encoder name: {encoder_name}")
            return False
    
    async def _ensure_encoders(self, encoder_names: List[str]) -> List[str]:
        """Проверяет доступность запрошенных энкодеров и вовращает только живые"""
        available = []
        
        for name in encoder_names:
            if await self._check_encoder_health(name):
                available.append(name)
        return available
    
    # ----------------------------------------------------------------------
    # Приватные методы: выполнение запросов к TEI
    # ----------------------------------------------------------------------
    
    async def _request_embed(
        self,
        encoder_name: str,
        inputs: Union[str, List[str]],
        prompt_type: Optional[str] = None,
        normalize: bool = True
    ) -> Optional[List[List[float]]]:
        """
        Выполняет запрос к TEI /embed endpoint

        Автоматически делит входной батчи на части в соответствии 
        с ограничением энкодера (у TEI есть max_client_batch_size)
        """
        client = self._get_http_client(encoder_name) 
        if not client:
            return None
        
        enc_info = await self._get_encoder_info(encoder_name)
        if not enc_info:
            return None
        
        # Будем учитывать ограничение конкретного TEI на размер батча
        max_batch_size = enc_info.max_client_batch_size
        
        try:
            url = f"{self._encoders[encoder_name]}{EPNames.EMBED}"

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
                    prompt_name=enc_info.prompt_names.get(prompt_type),
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
                        "Unexpected /tokenize response length: "
                        f"got {len(embeddings)} while expected {len(inputs_part)}"
                    )
                    return None
                
                embeddings_batch.extend(embeddings)
            
            return embeddings_batch
                
        except Exception as e:
            logger.error(f"TEI embed request to {encoder_name} failed: {e}")
            return None
        
    
    async def _request_tokenize(
        self,
        encoder_name: str,
        inputs: Union[str, List[str]]
    ) -> Optional[List[List[TokenInfo]]]:
        """
        Выполняет запрос к TEI /tokenize endpoint.

        Автоматически делит входной батчи на части в соответствии 
        с ограничением энкодера (у TEI есть max_client_batch_size)
        """

        client = self._get_http_client(encoder_name)
        if not client:
            return None
        
        enc_info = await self._get_encoder_info(encoder_name)
        if not enc_info:
            return None
        
        # Будем учитывать ограничение конкретного TEI на размер батча
        max_batch_size = enc_info.max_client_batch_size
        
        try:
            url = f"{self._encoders[encoder_name]}{EPNames.TOKENIZE}"

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
                        "Unexpected /tokenize response length: "
                        f"got {len(tokens)} while expected {len(inputs_part)}"
                    )
                    return None
                
                tokens_batch.extend(tokens)
            
            return tokens_batch
                
        except Exception as e:
            logger.error(f"TEI tokenize request to {encoder_name} failed: {e}")
            return None
    
    async def _request_multiple(
        self,
        encoder_names: List[str],
        request_func,
        **kwargs
    ) -> Dict[str, Optional[Any]]:
        """
        Выполняет параллельные запросы к нескольким энкодерам.

        """
        # Сначала выясним, какие сервисы из запрошенных "живы"
        available = await self._ensure_encoders(encoder_names)
        
        if not available:
            logger.warning("No encoders available for request")
            return {name: None for name in encoder_names}
        
        tasks = {}
        for name in available:
            task = asyncio.create_task(request_func(name, **kwargs))
            tasks[name] = task
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Request to {name} failed: {e}")
                results[name] = None
        
        for name in encoder_names:
            if name not in results:
                results[name] = None
        
        return results
    
    # ----------------------------------------------------------------------
    # Публичные методы
    # ----------------------------------------------------------------------
    
    async def encode_text(
        self,
        text: str,
        prompt_type: Optional[str] = PromptType.QUERY,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[List[float]]]:
        """Кодирует один текст в вектор."""
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        results = await self._request_multiple(
            encoder_names=use_encoders,
            request_func=self._request_embed,
            inputs=text,
            prompt_type=prompt_type,
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
        prompt_type: Optional[str] = PromptType.QUERY,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[List[List[float]]]]:
        """Пакетное кодирование текстов."""

        if not texts:
            return {encoder: None for encoder in (use_encoders or self._encoders)}

        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        results = await self._request_multiple(
            encoder_names=use_encoders,
            request_func=self._request_embed,
            inputs=texts,
            prompt_type=prompt_type,
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
            use_encoders = list(self._encoders.keys())
        
        tokenize_results: Dict[str, Optional[List[List[TokenInfo]]]] = await self._request_multiple(
            encoder_names=use_encoders,
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
            return {encoder: None for encoder in (use_encoders or self._encoders)}
        
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        tokenize_results: Dict[str, Optional[List[List[TokenInfo]]]] = await self._request_multiple(
            encoder_names=use_encoders,
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
    
    async def get_vector_size(self, encoder_name: str) -> Optional[int]:
        """Возвращает размерность вектора."""
        info = await self._get_encoder_info(encoder_name)
        return info.dimension if info else None
    
    async def get_vector_sizes(
        self,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[int]]:
        """
        Возвращает размерности векторов для нескольких энкодеров.
        """
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        # Создаем задачи для параллельного выполнения
        tasks = {
            name: self._get_encoder_info(name)
            for name in use_encoders
        }
        
        # Запускаем все задачи параллельно
        results = {}
        for name, task in tasks.items():
            try:
                info = await task
                results[name] = info.dimension if info else None
            except Exception as e:
                logger.error(f"Failed to get vector size for {name}: {e}")
                results[name] = None
        
        return results
    
    async def get_max_length(self, encoder_name: str) -> Optional[int]:
        """Возвращает максимальную длину текста в токенах."""
        info = await self._get_encoder_info(encoder_name)
        return info.max_input_length if info else None
    
    async def get_max_lengths(
        self,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[int]]:
        """
        Возвращает максимальные длины для нескольких энкодеров.
        """
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        # Создаем задачи для параллельного выполнения
        tasks = {
            name: self._get_encoder_info(name)
            for name in use_encoders
        }
        
        # Запускаем все задачи параллельно
        results = {}
        for name, task in tasks.items():
            try:
                info = await task
                results[name] = info.max_input_length if info else None
            except Exception as e:
                logger.error(f"Failed to get max length for {name}: {e}")
                results[name] = None
        
        return results
    
    async def get_model_name(self, encoder_name: str) -> Optional[str]:
        """Возвращает название модели."""
        info = await self._get_encoder_info(encoder_name)
        return info.model_id if info else None
    
    async def get_model_names(
        self,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[str]]:
        """
        Возвращает названия моделей для нескольких энкодеров.
        """
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        # Создаем задачи для параллельного выполнения
        tasks = {
            name: self._get_encoder_info(name)
            for name in use_encoders
        }
        
        # Запускаем все задачи параллельно
        results = {}
        for name, task in tasks.items():
            try:
                info = await task
                results[name] = info.model_id if info else None
            except Exception as e:
                logger.error(f"Failed to get model name for {name}: {e}")
                results[name] = None
        
        return results
    
    # ----------------------------------------------------------------------
    # Health checks
    # ----------------------------------------------------------------------
    
    async def health_check(self, encoder_name: str) -> bool:
        """Проверяет доступность конкретного энкодера."""
        client = self._get_http_client(encoder_name)
        if not client:
            return False
        
        try:
            url = f"{self._encoders[encoder_name]}/health"
            response = await client.get_with_retry(
                url=url,
                success_statuses={200}
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check for '{encoder_name}' failed: {e}")
            return False
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Проверяет доступность всех энкодеров."""
        tasks = {}
        for name in self._encoders:
            tasks[name] = asyncio.create_task(self.health_check(name))
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception:
                results[name] = False
        
        return results
    
    # ----------------------------------------------------------------------
    # Закрытие
    # ----------------------------------------------------------------------
    
    async def close(self):
        """Корректное закрытие всех HTTP клиентов."""
        close_tasks = []
        
        for client in self._http_clients.values():
            if client is not None:
                close_tasks.append(client.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self._http_clients = {name: None for name in self._encoders}
        logger.debug("EncoderClient closed")