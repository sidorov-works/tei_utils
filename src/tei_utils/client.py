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

import logging
logger = logging.getLogger(__name__)

# Структура InfoResponse содержит не все данные, 
# необходимые для работы клиента. В частности TEI не сообщают 
# в ответе /info важнейший параметр - размерность вектора. 
# Поэтому создадим отдельную pydantic модель EncoderInfo, 
# в которой данный параметр (dimension) будет присутствовать
class EncoderInfo(InfoResponse):
    dimension: Optional[int] = None

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
            base_timeout: float = 30.0,
            batch_timeout: float = 60.0
        ):
        """
        Инициализация клиента.

        Args:
            encoders: словарь, ключи которого являются именами TEI сервисов, 
                а значения - их URL.
            secret: секретный ключ для аутентификации в TEI сервисах
            base_timeout: тайм-аут на выполнение операций с одним текстом
            batch_timeout: тайм-аут на выполнение батчевых операций
        """

        self._secret = secret

        self._base_timeout = base_timeout
        self._batch_timeout = batch_timeout

        if encoders and isinstance(encoders, dict):
            self._encoders: Dict[str, str] = encoders.copy()    
        else:
            # Если не передали данные сервисов, то дальше инициализировать нечего
            logger.warning("EncoderClient initialized with no encoders")
            self._encoder_info = dict()
            self._base_clients = dict()
            self._batch_clients = dict()
            self._init_locks = dict()
            return
        
        # Сюда попадаем, если какие-то энкодеры переданы

        # Кэш метаданных энкодеров (тип InfoResponse)
        self._encoder_info: Dict[str, Optional[EncoderInfo]] = {
            name: None for name in self._encoders
        }
        
        # HTTP клиенты для каждого энкодера
        self._base_clients: Dict[str, Optional[RetryableHTTPClient]] = {
            name: None for name in self._encoders
        }
        self._batch_clients: Dict[str, Optional[RetryableHTTPClient]] = {
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
    
    async def _get_base_client(self, encoder_name: str) -> Optional[RetryableHTTPClient]:
        """Ленивое получение HTTP клиента для базовых запросов."""
        if encoder_name not in self._base_clients:
            logger.error(f"Unknown encoder: {encoder_name}")
            return None
        
        if self._base_clients[encoder_name] is None:
            client = RetryableHTTPClient(
                base_timeout=self._base_timeout,
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                total_timeout=self._base_timeout * 2
            )
            self._base_clients[encoder_name] = create_signed_client(
                client,
                secret=self._secret,
                service_name="encoder-client",
                auth_type=AuthType.SECRET_HEADER_AUTH
            )
        
        return self._base_clients[encoder_name]
    
    async def _get_batch_client(self, encoder_name: str) -> Optional[RetryableHTTPClient]:
        """Ленивое получение HTTP клиента для пакетных запросов."""
        if encoder_name not in self._batch_clients:
            logger.error(f"Unknown encoder: {encoder_name}")
            return None
        
        if self._batch_clients[encoder_name] is None:
            client = RetryableHTTPClient(
                base_timeout=self._batch_timeout,
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                total_timeout=self._batch_timeout * 2
            )
            self._batch_clients[encoder_name] = create_signed_client(
                client,
                secret=self._secret,
                service_name="encoder-client",
                auth_type=AuthType.SECRET_HEADER_AUTH
            )
        
        return self._batch_clients[encoder_name]
    
    # ----------------------------------------------------------------------
    # Приватные методы: получение информации об энкодерах
    # ----------------------------------------------------------------------
    
    async def _get_encoder_info(self, encoder_name: str) -> Optional[EncoderInfo]:
        """
        Получает основную информацию об энкодере `encoder_name` через `"/info"`,
        дополняет ее параметром dimension, который приходится выяснять 
        через отдельный вызов `"/embed"` (оригинальный *TEI* не включает 
        информацию о размерности вектора эмбеддинговой модели в ответ `"/info"`).

        Возвращает типизированный `EncoderInfo` - это `InfoResponse` с доп. атрибутом `dimension`.
        """
        if self._encoder_info.get(encoder_name) is not None:
            return self._encoder_info[encoder_name]
        
        if encoder_name not in self._encoders:
            logger.error(f"Encoder '{encoder_name}' not found in config")
            return None
        
        async with self._init_locks[encoder_name]:
            if self._encoder_info[encoder_name] is not None:
                return self._encoder_info[encoder_name]
            
            client = await self._get_base_client(encoder_name)
            if not client:
                return None
            
            try:
                info_url = f"{self._encoders[encoder_name]}/info"
                logger.debug(f"Fetching info from {encoder_name} at {info_url}")
                
                info_response = await client.get_with_retry(
                    url=info_url,
                    success_statuses={200}
                )
                
                data = info_response.json()
                # Запоминаем основные данные энкодера (пока без dimension)
                encoder_info = EncoderInfo(**data)

                # Теперь с помощью отдельного запроса /embed выясним размерность вектора
                embed_url = f"{self._encoders[encoder_name]}/embed"
                logger.debug(f"Fetching {encoder_name} vector dimension at {embed_url}")
                TEST_TEXT = "i"
                embed_request = EmbedRequest(TEST_TEXT)
                embed_response = await client.post_with_retry(
                    url=embed_url,
                    json=embed_request.model_dump(exclude_none=True),
                    headers={"Content-Type": "application/json"},
                    success_statuses={200}
                )
                embeddings = embed_response_adaptor.validate_json(embed_response.text)
                dimension = len(embeddings[0])
                encoder_info.dimension = dimension
                # Теперь информация о сервисе (энкодере) - полная: INFO_RESPONSE плюс dimension
                self._encoder_info[encoder_name] = encoder_info
                
                logger.info(f"✅ Encoder '{encoder_name}' initialized: {encoder_info.model_id}")
                return encoder_info
                
            except Exception as e:
                logger.warning(f"Failed to get info from encoder '{encoder_name}': {e}")
                return None
    
    async def _ensure_encoders(self, encoder_names: List[str]) -> List[str]:
        """Проверяет доступность запрошенных энкодеров и вовращает только живые"""
        available = []
        
        for name in encoder_names:
            # info = await self._get_encoder_info(name)
            # if info and info.status == "operational":
            #     available.append(name)
            # else:
            #     logger.warning(f"Encoder '{name}' is not available")
            pass # тут нужно вызывать /health, а не /info
        
        return available
    
    # ----------------------------------------------------------------------
    # Приватные методы: выполнение запросов к TEI с использованием моделей
    # ----------------------------------------------------------------------
    
    async def _request_embed(
        self,
        encoder_name: str,
        inputs: Union[str, List[str]],
        request_type: Optional[str] = None,
        normalize: bool = True,
        is_batch: bool = False
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """
        Выполняет запрос к TEI /embed endpoint
        """
        client = await (self._get_batch_client(encoder_name) if is_batch 
                       else self._get_base_client(encoder_name))
        
        if not client:
            return None
        
        try:
            url = f"{self._encoders[encoder_name]}/embed"
            
            # Создаём типизированный запрос
            request = EmbedRequest(
                inputs=inputs,
                prompt_name=request_type,
                normalize=normalize,
                truncate=True
            )
            
            response = await client.post_with_retry(
                url=url,
                json=request.model_dump(exclude_none=True),
                headers={"Content-Type": "application/json"},
                success_statuses={200}
            )
            
            # Парсим ответ через модель
            # embed_response = EmbedResponse(**response.json())
            embed_response = response # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # Извлекаем данные в зависимости от типа запроса
            if isinstance(inputs, str):
                return embed_response.embedding
            else:
                return embed_response.embeddings
                
        except Exception as e:
            logger.error(f"TEI embed request to {encoder_name} failed: {e}")
            return None
    
    async def _request_tokenize(
        self,
        encoder_name: str,
        inputs: Union[str, List[str]],
        is_batch: bool = False
    ) -> Optional[Union[int, List[int]]]:
        """
        Выполняет запрос к TEI /tokenize endpoint.
        Использует TokenizeRequest и TokenizeResponse модели.
        """
        client = await (self._get_batch_client(encoder_name) if is_batch 
                       else self._get_base_client(encoder_name))
        
        if not client:
            return None
        
        try:
            url = f"{self._encoders[encoder_name]}/tokenize"
            
            # Создаём типизированный запрос
            request = TokenizeRequest(
                inputs=inputs,
                add_special_tokens=True,
                truncate=True
            )
            
            response = await client.post_with_retry(
                url=url,
                json=request.model_dump(exclude_none=True),
                headers={"Content-Type": "application/json"},
                success_statuses={200}
            )
            
            # Парсим ответ через модель
            # tokenize_response = TokenizeResponse(**response.json())
            tokenize_response = response # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # Извлекаем данные в зависимости от типа запроса
            if isinstance(inputs, str):
                return tokenize_response.tokens_count
            else:
                return tokenize_response.tokens_counts
                
        except Exception as e:
            logger.error(f"TEI tokenize request to {encoder_name} failed: {e}")
            return None
    
    async def _request_multiple(
        self,
        encoder_names: List[str],
        request_func,
        **kwargs
    ) -> Dict[str, Optional[Any]]:
        """Выполняет параллельные запросы к нескольким энкодерам."""
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
    # Публичные методы (сигнатуры НЕ МЕНЯЮТСЯ)
    # ----------------------------------------------------------------------
    
    async def encode_text(
        self,
        text: str,
        request_type: Optional[str] = "query",
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[List[float]]]:
        """Кодирует один текст в вектор."""
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        results = await self._request_multiple(
            encoder_names=use_encoders,
            request_func=self._request_embed,
            inputs=text,
            request_type=request_type,
            normalize=True,
            is_batch=False
        )
        
        return results
    
    async def encode_batch(
        self,
        texts: List[str],
        request_type: Optional[str] = "query",
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[List[List[float]]]]:
        """Пакетное кодирование текстов."""
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        results = await self._request_multiple(
            encoder_names=use_encoders,
            request_func=self._request_embed,
            inputs=texts,
            request_type=request_type,
            normalize=True,
            is_batch=True
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
        
        results = await self._request_multiple(
            encoder_names=use_encoders,
            request_func=self._request_tokenize,
            inputs=text,
            is_batch=False
        )
        
        return results
    
    async def count_tokens_batch(
        self,
        texts: List[str],
        use_encoders: Optional[List[str]] = None
    ) -> Optional[List[Dict[str, Optional[int]]]]:
        """Подсчитывает количество токенов для нескольких текстов."""
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        if not texts:
            return []
        
        encoder_results = await self._request_multiple(
            encoder_names=use_encoders,
            request_func=self._request_tokenize,
            inputs=texts,
            is_batch=True
        )
        
        final_results = []
        for i in range(len(texts)):
            text_result = {}
            for encoder_name in use_encoders:
                encoder_counts = encoder_results.get(encoder_name)
                if encoder_counts and isinstance(encoder_counts, list) and i < len(encoder_counts):
                    text_result[encoder_name] = encoder_counts[i]
                else:
                    text_result[encoder_name] = None
            final_results.append(text_result)
        
        return final_results
    
    # ----------------------------------------------------------------------
    # Методы для получения информации (тоже используют типизированные ответы)
    # ----------------------------------------------------------------------
    
    async def get_vector_size(self, encoder_name: str) -> Optional[int]:
        """Возвращает размерность вектора."""
        info = await self._get_encoder_info(encoder_name)
        return info.dimension if info else None
    
    async def get_vector_sizes(
        self,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[int]]:
        """Возвращает размерности векторов для нескольких энкодеров."""
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        results = {}
        for name in use_encoders:
            info = await self._get_encoder_info(name)
            results[name] = info.dimension if info else None
        
        return results
    
    async def get_max_length(self, encoder_name: str) -> Optional[int]:
        """Возвращает максимальную длину текста в токенах."""
        info = await self._get_encoder_info(encoder_name)
        return info.max_input_length if info else None
    
    async def get_max_lengths(
        self,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[int]]:
        """Возвращает максимальные длины для нескольких энкодеров."""
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        results = {}
        for name in use_encoders:
            info = await self._get_encoder_info(name)
            results[name] = info.max_input_length if info else None
        
        return results
    
    async def get_model_name(self, encoder_name: str) -> Optional[str]:
        """Возвращает название модели."""
        info = await self._get_encoder_info(encoder_name)
        return info.model_id if info else None
    
    async def get_model_names(
        self,
        use_encoders: Optional[List[str]] = None
    ) -> Dict[str, Optional[str]]:
        """Возвращает названия моделей для нескольких энкодеров."""
        if use_encoders is None:
            use_encoders = list(self._encoders.keys())
        
        results = {}
        for name in use_encoders:
            info = await self._get_encoder_info(name)
            results[name] = info.model_id if info else None
        
        return results
    
    # ----------------------------------------------------------------------
    # Health checks
    # ----------------------------------------------------------------------
    
    async def health_check(self, encoder_name: str) -> bool:
        """Проверяет доступность конкретного энкодера."""
        client = await self._get_base_client(encoder_name)
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
        
        for client_dict in [self._base_clients, self._batch_clients]:
            for client in client_dict.values():
                if client is not None:
                    close_tasks.append(client.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self._base_clients = {name: None for name in self._encoders}
        self._batch_clients = {name: None for name in self._encoders}
        logger.debug("EncoderClient closed")


# Глобальный экземпляр
encoder_client = EncoderClient()