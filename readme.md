# TEI Utils

Асинхронный Python клиент для одновременных запросов к нескольким TEI-серверам (Hugging Face Text Embeddings Inference).

## Ключевая особенность

**Работа с несколькими TEI серверами одновременно** — клиент позволяет объединить несколько моделей эмбеддингов в одном интерфейсе и получать векторы от всех моделей параллельно за один вызов.

```python
# Один клиент — несколько моделей
client = EncoderClient(
    encoders={
        "bge-small": "http://localhost:8080",   # быстрая модель
        "bge-large": "http://localhost:8081",   # точная модель
        "e5-mistral": "http://localhost:8082"   # мультиязычная
    },
    secret="your-secret"
)

# Получаем эмбеддинги от всех моделей одновременно
vectors = await client.encode_text("Hello world")
# {
#     "bge-small": [0.1, 0.2, ...],   # 384-dim
#     "bge-large": [0.3, 0.4, ...],   # 1024-dim
#     "e5-mistral": [0.5, 0.6, ...]   # 4096-dim
# }
```

## Установка

```bash
pip install git+https://github.com/sidorov-works/tei_utils.git@v0.1.10
```

## Инициализация клиента

```python
from tei_utils import EncoderClient, PromptType

client = EncoderClient(
    # Словарь энкодеров: имя -> URL
    encoders={
        "bge-small": "http://localhost:8080",     # локальный TEI
        "bge-large": "http://localhost:8081",     # другой порт
        "e5-mistral": "https://tei.example.com"   # удаленный сервер
    },
    
    # Секретный ключ для Bearer аутентификации
    secret="your-secret-key",
    
    # Таймаут на один HTTP запрос (секунды)
    request_timeout=30.0,
    
    # Общий таймаут с учетом всех повторных попыток
    total_timeout=60.0
)
```

**Важно:**
- URL энкодеров должны указывать на корень TEI сервиса (например, `http://localhost:8080`)
- Все энкодеры используют единый секретный ключ для Bearer аутентификации
- Клиент автоматически запрашивает `/info` при первом обращении к энкодеру
- Информация об энкодере (размерность вектора, максимальная длина, доступные промпты) кэшируется

## Автоопределение промптов

**Клиент автоматически определяет правильные имена промптов для каждой модели** через эндпоинт `/info`.

Разные модели используют разные имена для query и document промптов:
- `BAAI/bge-*`: `search_query` / `search_document`
- `intfloat/e5-*`: `query` / `document`
- `snowflake/snowflake-arctic-embed-*`: `query` / `document`
- `ai-forever/FRIDA`: `search_query` / `search_document`

Клиент:
1. Запрашивает `/info` при первом обращении к энкодеру
2. Парсит поле `prompts` — список всех доступных промптов с их текстами
3. Фильтрует только промпты с непустым текстом
4. По имени находит подходящие для `query` и `document`
5. Кеширует найденные имена

```python
# Клиент сам подставит правильное имя промпта для каждой модели
query_vector = await client.encode_text(
    "search query",
    prompt_type=PromptType.QUERY      # 'query' для E5, 'search_query' для BGE
)

doc_vector = await client.encode_text(
    "document text", 
    prompt_type=PromptType.DOCUMENT   # 'document' для E5, 'search_document' для BGE
)
```

Если у модели нет промпта для указанного типа, запрос отправляется без `prompt_name`.

## Использование

### Кодирование текстов

```python
# Одиночный текст
result = await client.encode_text("What is machine learning?")
# {
#     "bge-small": [0.12, -0.34, ...],   # один вектор
#     "bge-large": [0.56, -0.78, ...]    # один вектор
# }

# Пакет текстов
texts = [
    "Machine learning is...",
    "Deep learning is...",
    "Neural networks..."
]
batch_result = await client.encode_batch(texts)
# {
#     "bge-small": [[...], [...], [...]],   # три вектора
#     "bge-large": [[...], [...], [...]]    # три вектора
# }

# С указанием типа промпта
# Клиент автоматически подставит правильное имя для каждой модели
query_vector = await client.encode_text(
    "search query",
    prompt_type=PromptType.QUERY     # автоопределение имени
)
doc_vector = await client.encode_text(
    "document text",
    prompt_type=PromptType.DOCUMENT  # автоопределение имени
)

# Можно передать явное имя промпта (минуя автоопределение)
custom_vector = await client.encode_text(
    "text",
    prompt_type="classification"     # напрямую передается в prompt_name
)
```

### Подсчет токенов

```python
# Одиночный текст
tokens = await client.count_tokens("Hello world")
# {"bge-small": 2, "bge-large": 2}

# Пакет текстов
tokens_batch = await client.count_tokens_batch(["Hello", "World", "!"])
# {"bge-small": [1, 1, 1], "bge-large": [1, 1, 1]}
```

### Работа с отдельными энкодерами

```python
# Получить информацию только для конкретной модели
dimension = await client.get_vector_size("bge-small")     # 384
max_length = await client.get_max_length("bge-small")     # 512
model_name = await client.get_model_name("bge-small")     # "BAAI/bge-small-en-v1.5"

# Проверка доступности
is_healthy = await client.health_check("bge-small")       # True/False
all_healthy = await client.health_check_all()             # {"bge-small": True, ...}

# Использовать только определенные энкодеры
vectors = await client.encode_text(
    "Hello",
    use_encoders=["bge-small"]  # только эта модель
)
```

### Обработка ошибок

```python
# Клиент возвращает None для недоступных энкодеров
vectors = await client.encode_text("Hello")
# {
#     "bge-small": [0.1, 0.2, ...],  # доступен
#     "bge-large": None              # недоступен
# }

# Проверяйте наличие результата
if vectors["bge-large"] is None:
    print("BGE Large is not available")
```

## Особенности

- 🔄 **Автоматическое определение промптов** — клиент сам узнает имена промптов у каждой модели через `/info`
- 🔄 **Автоматический батчинг** — клиент сам разбивает большие списки текстов на части, учитывая `max_client_batch_size` из `/info`
- ⚡ **Параллельные запросы** — при работе с несколькими энкодерами запросы выполняются одновременно
- 🔁 **Повторные попытки** — экспоненциальная задержка с jitter для сетевых ошибок и 5xx/429 статусов
- 💾 **Ленивая инициализация** — HTTP клиенты создаются только при первом обращении к энкодеру
- 🔐 **Bearer аутентификация** — автоматическое добавление `Authorization: Bearer <secret>` к каждому запросу
- 📝 **Pydantic валидация** — строгая типизация запросов и ответов TEI
- 🏥 **Health check перед запросами** — клиент проверяет доступность энкодеров перед отправкой

## Требования

- Python >= 3.9
- `httpx >= 0.28.1` — HTTP клиент
- `pydantic >= 2.12.5` — валидация данных
- `http-utils` — обертка с ретраями и аутентификацией

## Лицензия

MIT