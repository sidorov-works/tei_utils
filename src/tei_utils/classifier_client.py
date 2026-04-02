# src/tei_utils/classifier_client.py

"""
Клиент для работы с TEI-совместимыми сервисами классификации.
"""

from .tei_models import (
    PredictRequest,
    predict_single_response_adaptor,
    predict_batch_response_adaptor,
    LabelScore
)
from typing import List, Optional, Dict, Any, Union
from .tei_ep_names import EPNames
from .base_client import BaseClient

import logging
logger = logging.getLogger(__name__)


class ClassifierClient(BaseClient):
    """
    Клиент для работы с TEI-совместимыми сервисами классификации.
    
    При инициализации получает из конфига словарь {имя_классификатора: URL}.
    Для каждого классификатора лениво запрашивает /info при первом использовании.
    """
    
    # ----------------------------------------------------------------------
    # Приватные методы: выполнение запросов к TEI
    # ----------------------------------------------------------------------
    
    async def _request_predict(
        self,
        server_name: str,
        inputs: Union[str, List[str]],
        raw_scores: bool = False
    ) -> Optional[Union[List[LabelScore], List[List[LabelScore]]]]:
        """
        Выполняет запрос к TEI /predict endpoint.

        Автоматически делит входной батч на части в соответствии 
        с ограничением классификатора (max_client_batch_size).

        Args:
            server_name: Имя сервера (ключ из словаря servers)
            inputs: Текст (str) или список текстов (List[str]) для классификации
            raw_scores: Если True, возвращает raw logits вместо softmax вероятностей

        Returns:
            - Для одиночного текста: List[LabelScore]
            - Для батча: List[List[LabelScore]]
            - None при ошибке
        """
        client = self._get_http_client(server_name)
        if not client:
            return None
        
        base_info = await self._get_server_info(server_name)
        if not base_info:
            return None
        
        max_batch_size = base_info.max_client_batch_size
        
        # Определяем, одиночный текст или батч
        is_single = isinstance(inputs, str)
        
        # Для одиночного текста — отправляем как есть
        if is_single:
            try:
                url = f"{self._servers[server_name]}{EPNames.PREDICT}"
                request = PredictRequest(
                    inputs=inputs,
                    truncate=True,
                    raw_scores=raw_scores
                )
                
                response = await client.post_with_retry(
                    url=url,
                    json=request.model_dump(exclude_none=True),
                    headers={"Content-Type": "application/json"}
                )
                
                # Парсим через адаптер для одиночного текста
                result = predict_single_response_adaptor.validate_json(response.text)
                return result
                
            except Exception as e:
                logger.error(f"Predict request to {server_name} failed: {e}")
                return None
        
        # Для батча — разбиваем на части
        try:
            url = f"{self._servers[server_name]}{EPNames.PREDICT}"
            all_results = []
            
            for i in range(0, len(inputs), max_batch_size):
                inputs_part = inputs[i:i + max_batch_size]
                
                request = PredictRequest(
                    inputs=inputs_part,
                    truncate=True,
                    raw_scores=raw_scores
                )
                
                response = await client.post_with_retry(
                    url=url,
                    json=request.model_dump(exclude_none=True),
                    headers={"Content-Type": "application/json"}
                )
                
                # Парсим через адаптер для батча
                batch_results = predict_batch_response_adaptor.validate_json(response.text)
                
                if len(batch_results) != len(inputs_part):
                    logger.warning(
                        f"Unexpected {EPNames.PREDICT} response length: "
                        f"got {len(batch_results)} while expected {len(inputs_part)}"
                    )
                    return None
                
                all_results.extend(batch_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Predict request to {server_name} failed: {e}")
            return None
    
    # ----------------------------------------------------------------------
    # Публичные методы
    # ----------------------------------------------------------------------
    
    async def classify(
        self,
        text: str,
        raw_scores: bool = False,
        use_classifiers: Optional[List[str]] = None
    ) -> Dict[str, Optional[List[LabelScore]]]:
        """
        Классифицирует один текст.

        Args:
            text: Текст для классификации
            raw_scores: Если True, возвращает raw logits вместо softmax вероятностей
            use_classifiers: Список имен классификаторов для использования.
                Если None, используются все зарегистрированные серверы.

        Returns:
            Словарь {имя_классификатора: список LabelScore или None}
        """
        if use_classifiers is None:
            use_classifiers = list(self._servers.keys())
        
        results = await self._request_multiple(
            server_names=use_classifiers,
            request_func=self._request_predict,
            inputs=text,
            raw_scores=raw_scores
        )
        
        return results
    
    async def classify_batch(
        self,
        texts: List[str],
        raw_scores: bool = False,
        use_classifiers: Optional[List[str]] = None
    ) -> Dict[str, Optional[List[List[LabelScore]]]]:
        """
        Пакетная классификация нескольких текстов.

        Args:
            texts: Список текстов для классификации
            raw_scores: Если True, возвращает raw logits вместо softmax вероятностей
            use_classifiers: Список имен классификаторов для использования.
                Если None, используются все зарегистрированные серверы.

        Returns:
            Словарь {имя_классификатора: список списков LabelScore или None}
        """
        if not texts:
            return {classifier: None for classifier in (use_classifiers or self._servers)}
        
        if use_classifiers is None:
            use_classifiers = list(self._servers.keys())
        
        results = await self._request_multiple(
            server_names=use_classifiers,
            request_func=self._request_predict,
            inputs=texts,
            raw_scores=raw_scores
        )
        
        return results