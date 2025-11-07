# Структура проекта RAG Application

## Обзор

Это FastAPI-приложение для реализации RAG (Retrieval-Augmented Generation) системы с использованием векторной базы данных Qdrant. Проект предназначен для обработки документов различных форматов, их векторизации и последующего семантического поиска.

## Архитектура проекта

```
rag_parse/
├── app/                          # Основной код приложения
│   ├── __init__.py
│   ├── main.py                   # Точка входа FastAPI
│   ├── config/                   # Конфигурация
│   │   ├── __init__.py
│   │   └── settings.py           # Настройки приложения
│   ├── models/                   # Модели данных
│   │   ├── __init__.py
│   │   ├── document.py           # Модели для парсинга документов
│   │   └── schemas.py            # API схемы (Pydantic)
│   └── services/                 # Бизнес-логика
│       ├── __init__.py
│       ├── chunking_service.py   # Разбиение текста на чанки
│       ├── document_processor.py # Обработка документов (устаревший)
│       ├── embedding_service.py  # Генерация эмбеддингов
│       ├── file_parser.py        # Парсинг файлов (Excel, Word)
│       └── vector_store.py       # Работа с Qdrant
├── data/                         # Данные
│   └── sample_document.txt
├── docker-compose.yml            # Docker композиция
├── requirements.txt              # Python зависимости
└── README.md                     # Документация
```

---

## Детальное описание компонентов

### 1. `app/main.py` - Главное приложение FastAPI

**Назначение:** Точка входа в приложение, инициализация FastAPI и настройка middleware.

**Ключевые функции:**
- Создание экземпляра FastAPI с настройками из `settings.py`
- Настройка CORS middleware для разрешения кросс-доменных запросов
- Обработчики событий `startup` и `shutdown` для инициализации и очистки ресурсов
- Базовые эндпоинты здоровья (`/` и `/health`)

**Основные эндпоинты:**
- `GET /` - корневой эндпоинт, возвращает информацию о приложении
- `GET /health` - проверка состояния приложения

**Код на строках:** 1-65

**Будущие расширения:**
- Подключение роутеров для работы с документами и запросами (строки 61-64, закомментированы)

---

### 2. `app/config/settings.py` - Настройки приложения

**Назначение:** Централизованное управление конфигурацией через переменные окружения.

**Класс `Settings`:**
Использует `pydantic_settings.BaseSettings` для валидации и загрузки настроек из `.env` файла.

**Основные параметры:**

#### Конфигурация Qdrant:
- `qdrant_host: str = "localhost"` - адрес сервера Qdrant
- `qdrant_port: int = 6333` - порт Qdrant
- `qdrant_collection_name: str = "documents"` - название коллекции векторов

#### Модель эмбеддингов:
- `embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"` - модель для векторизации текста
  - Размерность: 384
  - Быстрая и эффективная модель для семантического поиска

#### API конфигурация:
- `api_title`, `api_version`, `api_description` - метаданные API
- `debug: bool = True` - режим отладки
- `log_level: str = "INFO"` - уровень логирования

#### CORS:
- `cors_origins: str` - разрешенные origins (разделенные запятой)
- `cors_origins_list` - свойство, преобразующее строку в список

**Код на строках:** 1-40

---

### 3. `app/models/document.py` - Модели для парсинга документов

**Назначение:** Pydantic модели для структурированного представления парсированных документов.

**Основные классы:**

#### `FileType(Enum)` (строки 7-12)
Перечисление поддерживаемых типов файлов:
- EXCEL - файлы .xlsx, .xls, .xlsm
- WORD - файлы .docx, .doc
- PDF - файлы .pdf
- TEXT - текстовые файлы

#### `DocumentChunk` (строки 14-26)
Представляет один чанк (фрагмент) документа:
- `content: str` - текстовое содержимое чанка
- `metadata: Dict[str, Any]` - метаданные (имя файла, тип, индекс строки и т.д.)
- `chunk_index: int` - порядковый номер чанка в документе
- Валидатор `content_not_empty` проверяет, что контент не пустой

#### `ParsedDocument` (строки 28-39)
Представляет полностью распарсенный документ:
- `filename: str` - имя исходного файла
- `file_type: FileType` - тип файла
- `chunks: List[DocumentChunk]` - список чанков
- `total_chunks: int` - общее количество чанков
- `file_size_bytes: Optional[int]` - размер файла
- `parsed_at: datetime` - временная метка парсинга

#### `ExcelMetadata` (строки 41-47)
Специфичные метаданные для Excel:
- `sheet_name: str` - название листа
- `row_number: int` - номер строки
- `column_names: List[str]` - названия колонок
- `total_rows: int` - общее количество строк

#### `WordMetadata` (строки 49-54)
Специфичные метаданные для Word:
- `paragraph_number: int` - номер параграфа
- `total_paragraphs: int` - общее количество параграфов
- `has_formatting: bool` - наличие специального форматирования (жирный, курсив, подчеркивание)

#### `FileValidationError` (строки 56-58)
Кастомное исключение для ошибок валидации файлов.

**Код на строках:** 1-58

---

### 4. `app/models/schemas.py` - API схемы

**Назначение:** Pydantic схемы для API запросов и ответов.

**Основные схемы:**

#### Документы:
- `DocumentBase` (строки 6-10) - базовая схема с контентом и метаданными
- `DocumentCreate` (строки 12-14) - для создания документа
- `DocumentResponse` (строки 17-24) - ответ с ID и временной меткой

#### Запросы поиска:
- `QueryRequest` (строки 26-36):
  - `query: str` - текст запроса (минимум 1 символ)
  - `top_k: int = 5` - количество результатов (1-20)
  - `score_threshold: float = 0.7` - минимальный порог сходства (0.0-1.0)

#### Результаты поиска:
- `SearchResult` (строки 38-44) - индивидуальный результат с ID, контентом, score и метаданными
- `QueryResponse` (строки 46-51) - ответ с результатами поиска

#### Загрузка файлов:
- `FileUploadResponse` (строки 53-58) - ответ при загрузке файла

**Код на строках:** 1-58

---

### 5. `app/services/file_parser.py` - Парсинг файлов

**Назначение:** Парсинг Excel и Word документов с детальной валидацией и сохранением структуры.

**Константы:**
- `MAX_FILE_SIZE = 50 MB` - максимальный размер файла
- `MAX_EXCEL_SIZE = 20 MB` - лимит для Excel
- `MAX_WORD_SIZE = 10 MB` - лимит для Word
- `EXCEL_EXTENSIONS = {'.xlsx', '.xls', '.xlsm'}`
- `WORD_EXTENSIONS = {'.docx', '.doc'}`

**Функции валидации:**

#### `validate_file_type(filename, allowed_extensions)` (строки 27-44)
- Проверяет расширение файла
- Выбрасывает `FileValidationError` для неподдерживаемых типов

#### `validate_file_size(file, max_size)` (строки 46-76)
- Проверяет размер файла
- Возвращает размер в байтах
- Выбрасывает исключение для слишком больших или пустых файлов

**Функции парсинга:**

#### `parse_excel(file, filename)` (строки 78-179)
Парсит Excel файлы с сохранением структуры:
1. Валидация типа и размера файла
2. Чтение всех листов через `pd.ExcelFile`
3. Для каждого листа:
   - Получение названий колонок
   - Обработка каждой строки
   - Сохранение данных в формате "ColumnName: Value | ColumnName: Value"
4. Создание `DocumentChunk` для каждой строки с метаданными:
   - Название листа
   - Номер строки
   - Названия колонок
   - Общее количество строк
5. Возвращает `ParsedDocument` с полной информацией

**Особенности:**
- Пропускает пустые листы и строки с NaN значениями
- Сохраняет контекст колонок для лучшего семантического поиска
- Детальные метаданные для каждого чанка

#### `parse_word(file, filename)` (строки 181-267)
Парсит Word документы:
1. Валидация типа и размера
2. Чтение через `python-docx`
3. Обработка каждого параграфа:
   - Пропуск пустых параграфов
   - Определение наличия форматирования (жирный/курсив/подчеркивание)
4. Создание чанков с метаданными параграфа
5. Возвращает `ParsedDocument`

#### `parse_file(file, filename)` (строки 269-293)
Универсальная функция-роутер:
- Определяет тип файла по расширению
- Вызывает соответствующий парсер
- Выбрасывает исключение для неподдерживаемых типов

**Код на строках:** 1-293

---

### 6. `app/services/chunking_service.py` - Разбиение на чанки

**Назначение:** Интеллектуальное разбиение текстов на фрагменты для эффективной векторизации.

**Класс `ChunkingService`:**

#### Инициализация (строки 13-44)
```python
ChunkingService(
    chunk_size=500,        # Максимальный размер чанка
    chunk_overlap=50,      # Перекрытие между чанками
    separators=None        # Сепараторы для разделения
)
```

Использует `RecursiveCharacterTextSplitter` из LangChain:
- Приоритетные сепараторы: `["\n\n", "\n", ". ", " ", ""]`
- Рекурсивный алгоритм для естественного разделения текста

#### `chunk_text(text, metadata)` (строки 46-86)
Разбивает произвольный текст на чанки:
- Пропускает пустой текст
- Создает `DocumentChunk` для каждого фрагмента
- Добавляет метаданные о методе чанкинга

#### `chunk_excel_by_rows(df, sheet_name, filename, rows_per_chunk=10)` (строки 88-161)
Специальный метод для Excel данных:
- Группирует строки по `rows_per_chunk`
- Включает заголовки колонок в каждый чанк для контекста
- Формат: `"Headers: col1 | col2 | col3\n\nData:\ncol1: val1 | col2: val2"`
- Сохраняет диапазоны строк в метаданных (start_row, end_row)

**Преимущество:** Поисковый запрос может находить данные по названиям колонок, даже если они в разных чанках.

#### `chunk_parsed_document(parsed_doc, rechunk, rows_per_chunk)` (строки 163-213)
Разбивает уже распарсенный документ:
- Для Excel: использует существующие чанки (rechunk требует исходный DataFrame)
- Для текстовых документов: опционально перечанкирует с новыми настройками
- Обновляет индексы чанков для последовательности

#### `update_chunk_size(chunk_size, chunk_overlap)` (строки 215-238)
Динамическое обновление параметров чанкинга:
- Пересоздает text_splitter с новыми настройками
- Полезно для оптимизации под разные типы документов

#### Глобальный экземпляр (строки 241-265)
```python
get_chunking_service(chunk_size=500, chunk_overlap=50)
```
Singleton паттерн для переиспользования сервиса.

**Код на строках:** 1-265

---

### 7. `app/services/embedding_service.py` - Генерация эмбеддингов

**Назначение:** Преобразование текста в векторные представления для семантического поиска.

**Класс `ChunkWithEmbedding` (строки 11-48):**
Контейнер для чанка с его эмбеддингом:
- `content: str` - текст
- `embedding: List[float]` - вектор эмбеддинга
- `metadata: Dict` - метаданные
- `chunk_index: int` - индекс
- Метод `to_dict()` для сериализации

**Класс `EmbeddingService`:**

#### Инициализация (строки 54-83)
```python
EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
- Загружает модель через `SentenceTransformer`
- Определяет размерность эмбеддинга (384 для all-MiniLM-L6-v2)
- Логирует информацию о модели

**Модель all-MiniLM-L6-v2:**
- Размер: 80 MB
- Размерность: 384
- Скорость: ~14000 предложений/сек
- Качество: хороший баланс скорость/качество

#### `embed_text(text)` (строки 84-108)
Генерирует эмбеддинг для одного текста:
- Обрабатывает пустой текст (возвращает нулевой вектор)
- Нормализация векторов для косинусного сходства
- Возвращает список float

#### `embed_batch(texts, batch_size=32, show_progress=False)` (строки 110-161)
Эффективная батч-генерация эмбеддингов:
- Фильтрует пустые тексты, сохраняя индексы
- Обрабатывает тексты батчами для оптимизации
- Опциональный прогресс-бар
- Вставляет нулевые векторы для пустых текстов

**Преимущество:** Батч-обработка в ~10-50 раз быстрее последовательной генерации.

#### `embed_chunks(chunks, batch_size=32, show_progress=True)` (строки 163-208)
Специализированный метод для `DocumentChunk`:
- Извлекает тексты из чанков
- Генерирует эмбеддинги батчем
- Создает `ChunkWithEmbedding` объекты
- Сохраняет метаданные

#### Вспомогательные методы:

**`compute_similarity(embedding1, embedding2)`** (строки 210-230)
- Вычисляет косинусное сходство
- Использует numpy для эффективных вычислений
- Возвращает значение 0-1

**`find_similar_chunks(query_embedding, chunks_with_embeddings, top_k=5)`** (строки 232-262)
- Находит наиболее похожие чанки
- Вычисляет сходство со всеми чанками
- Сортирует по убыванию сходства
- Возвращает top_k результатов

**`get_model_info()`** (строки 273-284)
Возвращает информацию о модели:
- Название модели
- Размерность эмбеддингов
- Максимальная длина последовательности

#### Глобальный экземпляр (строки 287-304)
```python
get_embedding_service(model_name=None)
```
Singleton для переиспользования загруженной модели.

**Код на строках:** 1-304

---

### 8. `app/services/vector_store.py` - Работа с Qdrant

**Назначение:** Интеграция с векторной базой данных Qdrant для хранения и поиска эмбеддингов.

**Класс `VectorStoreService`:**

#### Инициализация (строки 15-38)
```python
VectorStoreService(
    host="localhost",
    port=6333,
    collection_name="documents"
)
```
- Подключается к Qdrant клиенту
- Инициализирует embedding service
- Автоматически создает коллекцию если не существует

#### `_ensure_collection_exists()` (строки 40-55)
Создает коллекцию с конфигурацией:
- `VectorParams`:
  - `size` - размерность векторов (384)
  - `distance=Distance.COSINE` - метрика косинусного сходства

**Почему COSINE:**
- Инвариантна к длине вектора
- Оптимальна для normalized embeddings
- Диапазон: -1 (противоположные) до 1 (идентичные)

#### `add_document(content, metadata, doc_id)` (строки 57-94)
Добавляет один документ:
1. Генерирует UUID если ID не указан
2. Создает эмбеддинг текста
3. Формирует payload с контентом и метаданными
4. Создает `PointStruct` с ID, вектором и payload
5. Загружает в Qdrant через `upsert`

**Upsert:** Обновляет если существует, иначе создает.

#### `add_documents_batch(documents)` (строки 96-136)
Батч-загрузка документов:
- Принимает список словарей: `[{"content": str, "metadata": dict, "id": str}]`
- Генерирует эмбеддинги батчем (эффективно!)
- Создает список `PointStruct`
- Загружает все за один запрос

**Преимущество:** В 10+ раз быстрее отдельной загрузки документов.

#### `search(query, top_k=5, score_threshold=None)` (строки 138-174)
Семантический поиск:
1. Генерирует эмбеддинг запроса
2. Выполняет векторный поиск в Qdrant:
   - `limit=top_k` - количество результатов
   - `score_threshold` - минимальный порог сходства
3. Форматирует результаты:
   ```python
   {
       "id": str,
       "score": float,  # 0.0 - 1.0
       "content": str,
       "metadata": dict
   }
   ```

**Как работает поиск:**
- Qdrant использует HNSW (Hierarchical Navigable Small World) индекс
- Приблизительный поиск ближайших соседей
- Скорость: миллисекунды для миллионов векторов

#### `delete_document(doc_id)` (строки 176-191)
Удаляет документ по ID:
- Использует `points_selector` для указания ID
- Возвращает True при успехе

#### `get_collection_info()` (строки 193-206)
Возвращает статистику коллекции:
- Название
- Количество векторов
- Количество точек
- Статус

#### Глобальный экземпляр (строки 209-223)
```python
get_vector_store_service()
```
Singleton для переиспользования соединения.

**Код на строках:** 1-224

**Важная проблема на строке 7:**
```python
from app.services.embeddings import get_embedding_service
```
Неправильный импорт! Должно быть:
```python
from app.services.embedding_service import get_embedding_service
```

---

### 9. `app/services/document_processor.py` - Обработка документов (Устаревший)

**Статус:** Этот модуль устарел и дублирует функциональность `file_parser.py`.

**Назначение:** Базовый парсинг различных форматов документов.

**Класс `DocumentProcessor`:**

Содержит статические методы для обработки разных форматов:

#### `process_text_file(file_path)` (строки 14-31)
- Читает .txt файлы
- Возвращает весь контент одним чанком

#### `process_pdf_file(file_path)` (строки 33-54)
- Использует `pypdf.PdfReader`
- Извлекает текст постранично
- Возвращает список чанков (1 страница = 1 чанк)

#### `process_docx_file(file_path)` (строки 56-76)
- Использует `python-docx`
- Извлекает текст попараграфно
- Возвращает список чанков (1 параграф = 1 чанк)

#### `process_csv_file(file_path)` (строки 78-98)
- Читает через pandas
- Формат: "col1: val1 | col2: val2"
- 1 строка = 1 чанк

#### `process_excel_file(file_path)` (строки 100-124)
- Читает все листы
- Формат: "Sheet: name | col1: val1 | col2: val2"
- 1 строка = 1 чанк

#### `process_file(file_path)` (строки 126-171)
- Универсальный метод-роутер
- Определяет тип по расширению
- Создает словари с метаданными:
  ```python
  {
      "content": str,
      "metadata": {
          "filename": str,
          "file_type": str,
          "chunk_index": int,
          "total_chunks": int
      }
  }
  ```

**Отличия от file_parser.py:**
- Простая реализация без валидации
- Нет детальных метаданных (названия колонок, листов и т.д.)
- Нет Pydantic моделей
- Работает с путями к файлам, а не с file-like объектами

**Рекомендация:** Использовать `file_parser.py` для новой функциональности.

**Код на строках:** 1-171

---

## Взаимодействие компонентов

### Типичный flow обработки документа:

```
1. Загрузка файла через API
   ↓
2. file_parser.py: parse_file()
   - Валидация размера и типа
   - Парсинг с сохранением структуры
   - Возвращает ParsedDocument
   ↓
3. chunking_service.py: chunk_parsed_document()
   - Опционально перечанкирует
   - Для Excel: группирует строки с заголовками
   - Возвращает List[DocumentChunk]
   ↓
4. embedding_service.py: embed_chunks()
   - Батч-генерация эмбеддингов
   - Возвращает List[ChunkWithEmbedding]
   ↓
5. vector_store.py: add_documents_batch()
   - Загрузка в Qdrant
   - Возвращает список ID
```

### Типичный flow поиска:

```
1. Пользовательский запрос
   ↓
2. embedding_service.py: embed_text(query)
   - Генерация эмбеддинга запроса
   ↓
3. vector_store.py: search()
   - Векторный поиск в Qdrant
   - Возвращает похожие документы с scores
   ↓
4. Возврат результатов пользователю
```

---

## Технологический стек

### Backend:
- **FastAPI 0.104.1** - современный асинхронный web-фреймворк
- **Uvicorn 0.24.0** - ASGI сервер
- **Pydantic 2.5.3** - валидация данных и сериализация

### Векторная база данных:
- **Qdrant 1.7.0** - специализированная векторная БД
  - HNSW индексация
  - Поддержка фильтров
  - High throughput

### Embeddings:
- **Sentence-Transformers 2.2.2** - трансформеры для эмбеддингов
- **Модель:** all-MiniLM-L6-v2
  - Размерность: 384
  - Быстрая и качественная

### Обработка документов:
- **pandas 2.1.4** - работа с табличными данными
- **python-docx 1.1.0** - парсинг Word документов
- **pypdf 3.17.4** - парсинг PDF
- **openpyxl 3.1.2** - работа с Excel

### RAG фреймворк:
- **LangChain 0.1.0** - фреймворк для LLM приложений
  - RecursiveCharacterTextSplitter для чанкинга
- **langchain-community 0.0.10** - дополнительные интеграции

### DevOps:
- **Docker & Docker Compose** - контейнеризация
- **python-dotenv 1.0.0** - управление environment variables

---

## Конфигурация через Environment Variables

Приложение настраивается через `.env` файл:

```bash
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API
API_TITLE=RAG Application API
API_VERSION=1.0.0
DEBUG=True
LOG_LEVEL=INFO

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

**Преимущества:**
- Легко менять настройки без изменения кода
- Разные конфигурации для dev/staging/prod
- Безопасное хранение секретов

---

## Docker Compose Setup

### Сервисы:

#### 1. Qdrant (строки 4-16)
```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"  # REST API
    - "6334:6334"  # gRPC
  volumes:
    - ./qdrant_storage:/qdrant/storage
```

**Особенности:**
- Персистентное хранилище через volume
- Web UI доступен на http://localhost:6333/dashboard
- gRPC для высокопроизводительных операций

#### 2. FastAPI (строки 18-35)
```yaml
fastapi:
  build: .
  ports:
    - "8000:8000"
  volumes:
    - ./app:/app/app     # hot-reload
    - ./data:/app/data
  depends_on:
    - qdrant
  command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Особенности:**
- Автоперезагрузка при изменении кода (--reload)
- Ждет запуска Qdrant (depends_on)
- Environment variable `QDRANT_HOST=qdrant` для связи между контейнерами

### Сеть (строки 37-39)
- Bridge network `rag_network`
- Изолированная сеть для сервисов
- DNS resolution по именам сервисов

**Код на строках:** 1-43

---

## Текущие ограничения и известные проблемы

### 1. Импорт в vector_store.py (строка 7)
**Проблема:**
```python
from app.services.embeddings import get_embedding_service
```
**Должно быть:**
```python
from app.services.embedding_service import get_embedding_service
```

### 2. Нет API эндпоинтов
- Роутеры для документов и запросов не реализованы (main.py:61-64)
- Нужно создать:
  - `POST /api/v1/documents/upload` - загрузка файлов
  - `POST /api/v1/query` - поиск
  - `GET /api/v1/documents` - список документов
  - `DELETE /api/v1/documents/{id}` - удаление

### 3. Дублирование кода
- `document_processor.py` дублирует функциональность `file_parser.py`
- Рекомендуется удалить или рефакторить

### 4. Отсутствие error handling
- Нет централизованной обработки ошибок
- Нет custom exception handlers

### 5. Нет тестов
- Unit тесты
- Integration тесты
- Тестовые данные

### 6. Отсутствие аутентификации
- Нет защиты API
- Нет ограничения rate limit

### 7. Логирование
- Базовое логирование в stdout
- Нет structured logging
- Нет интеграции с monitoring системами

---

## Следующие шаги для развития

### 1. Реализовать API эндпоинты
```python
# app/api/documents.py
@router.post("/upload")
async def upload_document(file: UploadFile):
    # Parse -> Chunk -> Embed -> Store
    pass

# app/api/query.py
@router.post("/search")
async def search_documents(query: QueryRequest):
    # Embed -> Search -> Return results
    pass
```

### 2. Добавить полноценный RAG pipeline
- Интеграция с LLM (OpenAI, Anthropic, local models)
- Prompt templates
- Context assembly из найденных документов
- Response generation

### 3. Улучшить обработку документов
- Поддержка PDF через `file_parser.py`
- Поддержка изображений (OCR)
- Поддержка таблиц в PDF
- Markdown документы

### 4. Оптимизация поиска
- Гибридный поиск (dense + sparse)
- Metadata filtering
- Reranking результатов
- Query expansion

### 5. Мониторинг и обсервабельность
- Prometheus metrics
- Structured logging (JSON)
- OpenTelemetry traces
- Health checks для зависимостей

### 6. Тестирование
```python
# tests/test_file_parser.py
def test_parse_excel():
    # Test Excel parsing
    pass

# tests/test_embedding_service.py
def test_embed_batch():
    # Test batch embedding
    pass
```

### 7. Security
- JWT authentication
- API key management
- Rate limiting
- Input sanitization
- HTTPS в production

### 8. Production готовность
- Multi-stage Docker build
- Health checks в docker-compose
- Graceful shutdown
- Database migrations
- Backup strategy для Qdrant

---

## Производительность и масштабирование

### Текущие характеристики:

**Embeddings:**
- Model: all-MiniLM-L6-v2
- Скорость: ~14000 sentences/sec на GPU
- Батч-обработка: 32 текста за раз
- Оптимизация: нормализация векторов

**Qdrant:**
- HNSW индекс: O(log N) поиск
- In-memory storage для скорости
- Персистентность на диск
- Горизонтальное масштабирование через sharding

**Рекомендации для масштабирования:**

1. **Для большого объема документов (>1M):**
   - Qdrant кластер с репликацией
   - Distributed embedding generation
   - Redis cache для частых запросов

2. **Для высокой нагрузки (>1000 RPS):**
   - Multiple FastAPI instances за load balancer
   - Async processing через Celery/RQ
   - Connection pooling для Qdrant

3. **Оптимизация эмбеддингов:**
   - GPU acceleration для больших батчей
   - Model quantization (FP16 или INT8)
   - Batch size tuning под hardware

---

## Заключение

Проект представляет собой **solid foundation** для RAG системы с:

**Сильные стороны:**
- Четкая архитектура с разделением ответственности
- Использование современных best practices (Pydantic, FastAPI)
- Детальная обработка документов с сохранением структуры
- Эффективная батч-обработка эмбеддингов
- Docker-based deployment

**Готово для production:**
- Конфигурация через environment variables
- Структурированные логи
- Валидация данных
- Type hints

**Требует доработки:**
- Реализация API эндпоинтов
- Полноценный RAG pipeline с LLM
- Тестирование
- Мониторинг
- Аутентификация

**Архитектурное решение:**
Использование Singleton паттерна для сервисов (embedding, vector_store) оптимально для:
- Переиспользования тяжелых ресурсов (модели ML, соединения с БД)
- Предотвращения утечек памяти
- Упрощения dependency injection

Проект готов к расширению для production-ready RAG приложения.
