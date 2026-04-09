# src/tei_utils/base_client.py

"""
Клиент для работы с TEI-совместимыми сервисами эмбеддингов.
"""

from .tei_models import InfoResponse
from typing import List, Optional, Dict, Any, Union
import asyncio
from http_utils import RetryableHTTPClient, create_signed_client, AuthType
from .tei_ep_names import EPNames

from typing import TypeVar, Callable, Awaitable 
T = TypeVar('T') # для обозначения типа результата _request_multiple()

import logging
logger = logging.getLogger(__name__)


class BaseClient:
    """
    Клиент для работы с TEI-совместимыми сервисами эмбеддингов.
    
    При инициализации получает из конфига словарь {имя_сервера: URL}.
    Для каждого сервера лениво запрашивает /info при первом использовании.
    
    """
    
    def __init__(
            self,
            servers: Dict[str, str],
            secret: Optional[str] = None,
            request_timeout: float = 60.0
        ):
        """
        Инициализация клиента.

        Args:
            servers: словарь, ключи которого являются именами TEI сервисов, 
                а значения - их URL.

            secret: секретный ключ для аутентификации в TEI сервисах

            request_timeout: тайм-аут на выполнение запроса к TEI (без учета ретраев)
        """

        self._secret = secret

        self._request_timeout = request_timeout

        if servers and isinstance(servers, dict):
            self._servers: Dict[str, str] = servers.copy()    
        else:
            # Если не передали данные сервисов, то дальше инициализировать нечего
            logger.warning("TEI Client initialized with no servers")
            self._server_info = dict()
            self._http_clients = dict()
            self._init_locks = dict()
            return
        
        # Сюда попадаем, если какие-то серверы переданы

        # Кэш метаданных серверов
        self._server_info: Dict[str, Optional[InfoResponse]] = {
            name: None for name in self._servers
        }
        
        # HTTP клиенты для каждого сервера
        self._http_clients: Dict[str, Optional[RetryableHTTPClient]] = {
            name: None for name in self._servers
        }
        
        # Блокировки для безопасной ленивой инициализации
        self._init_locks: Dict[str, asyncio.Lock] = {
            name: asyncio.Lock() for name in self._servers
        }
        
        logger.debug(f"TEI Client initialized with servers: {list(self._servers.keys())}")
    
    # ----------------------------------------------------------------------
    # Приватные методы: инициализация HTTP клиентов
    # ----------------------------------------------------------------------
    
    def _get_http_client(self, server_name: str) -> Optional[RetryableHTTPClient]:
        """
        Ленивое получение HTTP клиента для базовых запросов.

        TEI использует базовую аутентификацию запросов:
        `auth_header_name: str = "Authorization"`, `auth_header_scheme: str = "Bearer"`

        Возвращает None, если имя server_name не знакомо клиенту.
        """
        # Возвращаем None, если имя server_name не знакомо
        if server_name not in self._http_clients:
            logger.error(f"Unknown server: {server_name}")
            return None
        
        if self._http_clients[server_name] is None:
            client = RetryableHTTPClient(
                base_timeout=self._request_timeout,
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                total_timeout=self._request_timeout * 1.5
            )
            # Оборачиваем http клиента для автоматического подписания запросов 
            # только в том случае, если при создании клиента был передан secret
            if self._secret:
                self._http_clients[server_name] = create_signed_client(
                    client,
                    secret=self._secret,
                    service_name="TEI-client",
                    auth_type=AuthType.SECRET_HEADER_AUTH
                )
            else:
                self._http_clients[server_name] = client
        
        return self._http_clients[server_name]
    
    
    # ----------------------------------------------------------------------
    # Приватные методы: получение информации об серверах
    # ----------------------------------------------------------------------
    
    async def _get_server_info(self, server_name: str) -> Optional[InfoResponse]:
        """
        Получает основную информацию об сервере `server_name` через `"/info"`
        """
        logger.debug("In _get_server_info()")
        if self._server_info.get(server_name) is not None:
            return self._server_info[server_name]
        
        if server_name not in self._servers:
            logger.error(f"Server '{server_name}' not found in config")
            return None
        
        async with self._init_locks[server_name]:
            
            client = self._get_http_client(server_name)
            if not client:
                return None
            
            try:
                info_url = f"{self._servers[server_name]}{EPNames.INFO}"
                logger.debug(f"Fetching info from {server_name} at {info_url}")
                
                info_response = await client.get_with_retry(
                    url=info_url,
                    success_statuses={200}
                )
                
                data = info_response.json()
                # Запоминаем основные данные сервера 
                server_info = InfoResponse.model_validate(data)

                self._server_info[server_name] = server_info
                
                logger.info(f"✅ Server '{server_name}' initialized: {server_info.model_id}")
                return server_info
                
            except Exception as e:
                logger.warning(f"Failed to get info from server '{server_name}': {e}")
                return None
            
    async def _check_server_health(self, server_name: str) -> bool:
        """Проверка здоровья одного сервера"""
        http_client = self._get_http_client(server_name)
        if http_client:
            try:
                health_url = f"{self._servers[server_name]}{EPNames.HEALTH}"
                _ = await http_client.get_with_retry(health_url)
                return True
            except:
                logger.debug(f"Failed to check server {server_name} /health")
                return False
        else:
            logger.debug(f"Unknown server name: {server_name}")
            return False
        
    async def _check_servers_health(self, server_names: List[str]) -> Dict[str, bool]:
        """
        Проверяет здоровье указанных серверов параллельно.

        Returns:
            Словарь {имя_сервера: True/False}
        """
        tasks = {}
        for name in server_names:
            if name not in self._servers:
                continue
            tasks[name] = asyncio.create_task(self._check_server_health(name))
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception:
                results[name] = False
        
        return results
    
    async def _ensure_servers(self, server_names: List[str]) -> List[str]:
        """Проверяет доступность серверов параллельно, возвращает список живых."""
        health_results = await self._check_servers_health(server_names)
        return [name for name, is_healthy in health_results.items() if is_healthy]
    
    # ----------------------------------------------------------------------
    # Приватные методы: выполнение запросов к TEI
    # ----------------------------------------------------------------------
    
    async def _request_multiple(
        self,
        server_names: List[str],
        request_func: Callable[[str, Any], Awaitable[T]],
        **kwargs
    ) -> Dict[str, Optional[T]]:
        """
        Выполняет параллельные запросы к нескольким серверам.

        """
        # Сначала выясним, какие сервисы из запрошенных "живы"
        available = await self._ensure_servers(server_names)
        
        if not available:
            logger.warning("No servers available for request")
            return {name: None for name in server_names}
        
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
        
        for name in server_names:
            if name not in results:
                results[name] = None
        
        return results
    
    # ----------------------------------------------------------------------
    # Методы для получения информации 
    # ----------------------------------------------------------------------

    async def get_max_length(self, server_name: str) -> Optional[int]:
        """Возвращает максимальную длину текста в токенах."""
        info = await self._get_server_info(server_name)
        return info.max_input_length if info else None
    
    async def get_max_lengths(
        self,
        use_servers: Optional[List[str]] = None
    ) -> Dict[str, Optional[int]]:
        """
        Возвращает максимальные длины для нескольких серверов.
        """
        if use_servers is None:
            use_servers = list(self._servers.keys())
        
        # Создаем задачи для параллельного выполнения
        tasks = {
            name: self._get_server_info(name)
            for name in use_servers
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
    
    async def get_model_name(self, server_name: str) -> Optional[str]:
        """Возвращает название модели."""
        info = await self._get_server_info(server_name)
        return info.model_id if info else None
    
    async def get_model_names(
        self,
        use_servers: Optional[List[str]] = None
    ) -> Dict[str, Optional[str]]:
        """
        Возвращает названия моделей для нескольких серверов.
        """
        if use_servers is None:
            use_servers = list(self._servers.keys())
        
        # Создаем задачи для параллельного выполнения
        tasks = {
            name: self._get_server_info(name)
            for name in use_servers
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
    
    async def health_check(self, server_name: str) -> bool:
        """Проверяет доступность конкретного сервера."""
        return await self._check_server_health(server_name)
        
    async def health_check_all(self) -> Dict[str, bool]:
        """Проверяет доступность всех серверов параллельно."""
        return await self._check_servers_health(list(self._servers.keys()))
    
    
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
        
        self._http_clients = {name: None for name in self._servers}
        logger.debug("TEI Client closed")