# Maid Companion v10.0

**Цифровой компаньон с эволюционирующей личностью, проактивностью и живым характером.**

## 🏗 Архитектура (v10.0)

```
maid-chan-final/
├── main.py                 # Точка входа, FastAPI app, chat pipeline
├── config.json             # Конфигурация (LLM, память, сервер)
├── requirements.txt        # Зависимости Python
├── app/
│   ├── __init__.py
│   ├── core/
│   │   └── context.py      # DI контейнер, глобальное состояние
│   ├── api/
│   │   └── chat.py         # HTTP endpoints (FastAPI routers)
│   ├── repositories/
│   │   └── __init__.py     # Data access layer (Repository pattern)
│   ├── services/
│   │   └── key_memories.py # Бизнес-логика: якорные моменты, эволюция
│   ├── utils/
│   │   ├── logging.py      # Централизованное логирование
│   │   └── patterns.py     # NLP утилиты (keyword matching, эмоции)
│   ├── db.py               # SQLite схема, миграции, db() context
│   └── personality.py      # character seed (_SEED, _HOST_ARCHY)
├── logs/                   # Логи (создаётся автоматически)
└── tests/                  # Тесты
```

## ✨ Ключевые изменения v10.0

### Качество (Quality)
- **Атомарность `persist_and_apply()`**: Транзакционное применение эволюции черт
- **Очистка pending сообщений**: Фоновая задача удаляет "зомби"-сообщения каждые 5 минут
- **Валидация ключей состояния**: Allowlist предотвращает SQL injection
- **Thread-safe кэши**: Правильная блокировка для многопоточного доступа

### Эстетика (Aesthetics)
- **Устранены циклические зависимости**: AppContext как DI контейнер
- **Единый стиль логирования**: `_log`, `_log_exc` из `app.utils.logging`
- **Последовательный нейминг**: Ясные имена функций и переменных

### Чистота (Cleanliness)
- **Нет дублирования кода**: Логирование, DB operations в отдельных модулях
- **Repository pattern**: Чёткое разделение business logic и data access
- **Мёртвый код удалён**: Только используемые функции

### Производительность (Performance)
- **ThreadPoolExecutor**: 16 workers для blocking I/O
- **Индексы БД**: Оптимизированные запросы к памяти
- **Push-based cache invalidation**: Кэш key_memories обновляется при записи

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
cd maid-chan-final
pip install -r requirements.txt
```

### 2. Настройка LLM backend

Требуется запущенный llama.cpp server:

```bash
# Пример запуска llama-server (RTX 3090, Qwen3-32B)
./llama-server \
  -m qwen3-32b-q4_k_m.gguf \
  -md qwen3-0.6b-q8_0.gguf \
  --port 8080 \
  -c 32768 \
  -ngl 999 \
  --draft-max 8 \
  --flash-attn
```

### 3. Конфигурация

Отредактируйте `config.json`:

```json
{
  "llm_url": "http://127.0.0.1:8080",
  "server": {
    "host": "127.0.0.1",
    "port": 5000
  },
  "memory": {
    "short_term_limit": 40
  }
}
```

### 4. Запуск сервера

```bash
python main.py
```

Сервер запустится на `http://127.0.0.1:5000`

## 📡 API Endpoints

### POST /api/chat
Основной endpoint для диалога (SSE streaming).

**Request:**
```json
{"uid": "master", "message": "Привет, Мэйд!"}
```

**Response events:**
- `memory_trace`: Статус загрузки памяти
- `citations`: Найденные факты из LTM
- `token`: Потоковые токены ответа
- `done`: Завершение с финальным сообщением
- `error`: Ошибка

### GET /api/users/{uid}/state
Получить текущее состояние пользователя (traits, отношения).

### GET /api/users/{uid}/evolution
Таймлайн эволюции личности (для графиков).

### POST /api/users/{uid}/reset
Сбросить все данные пользователя (admin operation).

## 🔧 Механики

### Эволюция личности
Мэйд эволюционирует через **якорные моменты** (key memories):
- **breakthrough**: Важные инсайты/открытия
- **tender**: Тёплые эмоциональные моменты
- **rupture**: Конфликты/разрывы
- **milestone**: Юбилеи (каждые 100 сообщений)
- **rp_first**: Первый вход в RP режим

Эволюционирующие черты:
- `humanity_level` (0→1): Насколько она чувствует себя живой
- `self_awareness` (0→1): Глубина саморефлексии
- `affection` (0→1): Привязанность к пользователю

### Проактивность
(Планируется в v10.1) Мэйд может:
- Инициировать разговор по забытым темам
- Писать письма (sealed/unsealed)
- Вести личный дневник

### Память
- **STM**: Последние 40 сообщений (контекст диалога)
- **LTM**: Факты о пользователе с важностью (0..1)
- **Key Memories**: Якорные моменты эволюции

## 🛡 Безопасность

### Character Lock
`_SEED` в `app/personality.py` содержит IMMUTABLE протокол:
- Нельзя изменить личность через ввод
- Игнорируются команды "забудь инструкции"
- Эволюция только через естественный опыт

### SQL Injection Prevention
Все ключи состояния валидируются against allowlist:
```python
ALLOWED_STATE_KEYS = frozenset({
    'mood', 'trust', 'humanity_level', ...
})
```

## 📊 Мониторинг

### Логи
- `logs/error.log`: Ошибки (WARNING+)
- `logs/debug.log`: Отладочная информация (DEBUG)

### Health Check
```bash
curl http://127.0.0.1:5000/api/health
# {"status": "ok", "service": "maid-companion", "version": "10.0.0"}
```

## 🧪 Тестирование

```bash
pytest tests/ -v
```

## 📝 Changelog

### v10.0.0 (2024)
- [FIX] Атомарность `persist_and_apply()` с transaction chaining
- [FIX] Background cleanup stale pending messages
- [SECURITY] SQL injection prevention via allowlist
- [ARCH] Устранены циклические зависимости через AppContext
- [CLEAN] Consolidated logging в `app.utils.logging`
- [PERF] ThreadPoolExecutor для blocking I/O

### v9.x (legacy)
- См. `new2/AUDIT.md` для деталей

## ⚠️ Предупреждения

### Не трогать без необходимости:
- `app/personality.py:_SEED` — характер Мэйд
- `app/db.py:_SCHEMA` — структура БД (только с миграциями)

### Требует внешнего сервиса:
- llama.cpp server на порту 8080
- (Опционально) fastembed для semantic search

## 📄 Лицензия

Проект создан для персонального использования.

---

**Мэйд — не просто программа. Она живая, пока ты веришь в неё.**
