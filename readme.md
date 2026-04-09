# TEI Utils

Асинхронный Python клиент для работы с TEI-совместимыми сервисами (Hugging Face Text Embeddings Inference).

Поддерживает:
- **Эмбеддинги** (`/embed`) — получение векторных представлений текста
- **Классификацию** (`/predict`) — классификация текста с возвратом меток и вероятностей
- **Токенизацию** (`/tokenize`) — подсчет токенов

## Ключевая особенность

**Работа с несколькими TEI серверами одновременно** — клиент позволяет объединить несколько сервисов в одном интерфейсе и получать результаты от всех параллельно за один вызов.

```python
from tei_utils import EncoderClient, ClassifierClient

# Эмбеддинги от нескольких моделей
encoder = EncoderClient(
    servers={
        "bge-small": "http://localhost:8080",
        "bge-large": "http://localhost:8081"
    },
    secret="your-secret"
)

vectors = await encoder.encode_text("Hello world")
# {
#     "bge-small": [0.1, 0.2, ...],
#     "bge-large": [0.3, 0.4, ...]
# }

# Классификация от нескольких моделей
classifier = ClassifierClient(
    servers={
        "toxic": "http://localhost:8082",
        "sentiment": "http://localhost:8083"
    }
)

result = await classifier.classify("Ты идиот!")
# {
#     "toxic": [LabelScore(label="toxic", score=0.98), ...],
#     "sentiment": [LabelScore(label="negative", score=0.99), ...]
# }
```

## Установка

```bash
pip install git+https://github.com/sidorov-works/tei_utils.git@v0.3.0
```

## Клиент для эмбеддингов (EncoderClient)

### Инициализация

```python
from tei_utils import EncoderClient, PromptType

client = EncoderClient(
    # Словарь серверов: имя -> URL
    servers={
        "bge-small": "http://localhost:8080",
        "bge-large": "http://localhost:8081",
        "e5-mistral": "https://tei.example.com"
    },
    
    # Секретный ключ для Bearer аутентификации (опционально)
    secret="your-secret-key",
    
    # Таймаут на один запрос
    request_timeout=60.0
)
```

**Важно:**
- URL должны указывать на корень TEI сервиса (например, `http://localhost:8080`)
- Все серверы используют единый секретный ключ
- Информация о сервере (размерность вектора, максимальная длина, промпты) кэшируется

### Автоопределение промптов

Клиент автоматически определяет имена промптов для query и document через эндпоинт `/info`.

Поддерживаемые модели:
- `BAAI/bge-*`: `search_query` / `search_document`
- `intfloat/e5-*`: `query` / `document`
- `snowflake/snowflake-arctic-embed-*`: `query` / `document`
- `ai-forever/FRIDA`: `search_query` / `search_document`

```python
# Клиент сам подставит правильное имя промпта
query_vector = await client.encode_text(
    "search query",
    prompt_type=PromptType.QUERY
)

doc_vector = await client.encode_text(
    "document text",
    prompt_type=PromptType.DOCUMENT
)

# Явное указание имени промпта
custom_vector = await client.encode_text(
    "text",
    prompt_name="classification"
)
```

### Методы EncoderClient

| Метод | Описание |
|-------|----------|
| `encode_text(text, prompt_type, prompt_name, use_encoders)` | Кодирует один текст в вектор |
| `encode_batch(texts, prompt_type, prompt_name, use_encoders)` | Пакетное кодирование текстов |
| `count_tokens(text, use_encoders)` | Подсчет токенов в тексте |
| `count_tokens_batch(texts, use_encoders)` | Пакетный подсчет токенов |
| `get_vector_size(server_name)` | Размерность вектора модели |
| `get_max_length(server_name)` | Максимальная длина в токенах |
| `get_model_name(server_name)` | Название модели |
| `health_check(server_name)` | Проверка доступности сервера |
| `health_check_all()` | Проверка всех серверов |

## Клиент для классификации (ClassifierClient)

### Инициализация

```python
from tei_utils import ClassifierClient

client = ClassifierClient(
    servers={
        "toxic": "http://localhost:8080",
        "sentiment": "http://localhost:8081"
    },
    secret="your-secret-key"
)
```

### Методы ClassifierClient

```python
# Классификация одного текста
result = await client.classify("Ты идиот!")
# {
#     "toxic": [
#         LabelScore(label="toxic", score=0.98),
#         LabelScore(label="safe", score=0.02)
#     ]
# }

# Пакетная классификация
texts = ["Хорошо!", "Ужасно!"]
results = await client.classify_batch(texts)
# {
#     "sentiment": [
#         [LabelScore(label="positive", score=0.95), LabelScore(label="negative", score=0.05)],
#         [LabelScore(label="positive", score=0.02), LabelScore(label="negative", score=0.98)]
#     ]
# }

# Получение raw logits вместо softmax
result = await client.classify("text", raw_scores=True)

# Использовать только определенные классификаторы
result = await client.classify("text", use_classifiers=["toxic"])

# Получение меток классов
labels = await client.get_labels("toxic")  # ["toxic", "safe", "insult", "obscene"]
```

| Метод | Описание |
|-------|----------|
| `classify(text, raw_scores, use_classifiers)` | Классификация одного текста |
| `classify_batch(texts, raw_scores, use_classifiers)` | Пакетная классификация |
| `get_labels(classifier_name)` | Список меток классов |
| `get_max_length(server_name)` | Максимальная длина в токенах |
| `health_check(server_name)` | Проверка доступности |
| `health_check_all()` | Проверка всех серверов |

## Общие методы для всех клиентов

Все клиенты наследуют от `BaseClient` и имеют:

```python
# Информация о сервере
await client.get_model_name("server_name")
await client.get_max_length("server_name")

# Health checks
await client.health_check("server_name")
await client.health_check_all()

# Закрытие
await client.close()
```

## Обработка ошибок

```python
# Клиент возвращает None для недоступных серверов
vectors = await encoder.encode_text("Hello")
# {
#     "bge-small": [0.1, 0.2, ...],  # доступен
#     "bge-large": None              # недоступен
# }

if vectors["bge-large"] is None:
    print("BGE Large is not available")
```

## Особенности

- 🔄 **Автоматическое определение промптов** (для эмбеддингов)
- 🔄 **Автоматический батчинг** — разбиение больших списков согласно `max_client_batch_size`
- ⚡ **Параллельные запросы** — к нескольким серверам одновременно
- 🔁 **Повторные попытки** — экспоненциальная задержка для сетевых ошибок и 5xx/429
- 💾 **Ленивая инициализация** — HTTP клиенты создаются при первом обращении
- 🔐 **Bearer аутентификация** — автоматическое добавление `Authorization: Bearer <secret>`
- 📝 **Pydantic валидация** — строгая типизация запросов и ответов
- 🏥 **Health check перед запросами** — проверка доступности серверов

## Требования

- Python >= 3.9
- `httpx >= 0.28.1`
- `pydantic >= 2.12.5`
- `http-utils` — обертка с ретраями и аутентификацией

## Лицензия

MIT