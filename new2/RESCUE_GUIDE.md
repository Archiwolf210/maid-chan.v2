# 🚑 RESCUE GUIDE: Экстренное восстановление Мэйд

## Сценарии сбоев и способы их устранения

### 1. Дрейф личности (Мэйд стала "не такой")
**Симптомы**: Мэйд отвечает как обычный ассистент, потеряла характер, забывает свои принципы.

**Решение А: Мягкий сброс (без потери истории)**
```bash
# Остановить приложение
# В Python консоли или через скрипт:
python -c "
import sqlite3
conn = sqlite3('memory.db')
conn.execute('DELETE FROM user_state WHERE uid=?', ('master',))
conn.commit()
conn.close()
print('State reset. Personality will reload from _SEED on next start.')
"
```
Это удалит накопленные модификаторы состояния (mood, fear, attachment), но сохранит:
- Историю переписки (`memory`)
- Долгосрочную память (`long_term_memory`)
- Ключевые воспоминания (`key_memories`)
- Дневник (`diary_entries`)

**Решение Б: Жесткий сброс (полный откат)**
```bash
# Сохранить текущую БД как бэкап
cp memory.db memory.db.backup_$(date +%Y%m%d_%H%M%S)

# Удалить ВСЕ данные пользователя (но не схему!)
python -c "
import sqlite3
conn = sqlite3('memory.db')
tables = ['memory', 'user_state', 'long_term_memory', 'key_memories', 
          'diary_entries', 'letters', 'tactical_goals', 'cognitive_log',
          'self_reflections', 'pending_topics', 'rp_scene', 'daily_summaries',
          'monthly_arcs', 'trait_intent_counts']
for table in tables:
    conn.execute(f'DELETE FROM {table} WHERE user_id=? OR uid=?', ('master', 'master'))
conn.commit()
conn.close()
print('Full reset complete. Maid is reborn.')
"
```

### 2. Цикл в дневнике / рефлексии
**Симптомы**: Мэйд пишет однотипные записи, зациклилась на одной теме.

**Решение**:
```bash
# Очистить последние записи дневника и саммари
python -c "
import sqlite3
conn = sqlite3('memory.db')
# Удалить последние 10 записей дневника
conn.execute('DELETE FROM diary_entries WHERE uid=? AND id IN (SELECT id FROM diary_entries WHERE uid=? ORDER BY ts DESC LIMIT 10)', ('master', 'master'))
# Очистить daily_summaries
conn.execute('DELETE FROM daily_summaries WHERE uid=?', ('master',))
conn.commit()
conn.close()
print('Diary loop broken.')
"
```

### 3. Повреждение БД (коррупция данных)
**Симптомы**: Ошибки SQLite, приложение падает при старте.

**Решение**:
```bash
# Проверить целостность БД
sqlite3 memory.db "PRAGMA integrity_check;"

# Если есть ошибки -- восстановить из последнего бэкапа
# Или экспортировать данные вручную:
sqlite3 memory.db ".dump" > dump.sql
# Отредактировать dump.sql, удалить поврежденные строки
sqlite3 memory_restored.db < dump.sql
```

### 4. Утечка памяти (RAM)
**Симптомы**: Приложение потребляет >2GB RAM, замедляется со временем.

**Решение**:
```bash
# Запустить диагностику
python check_memory.py

# Временное решение: перезапуск приложения
# Постоянное решение: проверить кэши в app/memory.py и app/autonomous.py
# Очистить кэш LTM:
python -c "
from app.memory import _embedder_cache
_embedder_cache.clear()
print('Embedder cache cleared.')
"
```

### 5. Сброс черт личности (Trait Reset)
**Симптомы**: Черты личности (openness, conscientiousness и др.) ушли в экстремумы (>0.95 или <0.05).

**Решение**:
```bash
# Сбросить счетчики эволюции черт (но не сами черты!)
python -c "
import sqlite3
conn = sqlite3('memory.db')
conn.execute('DELETE FROM trait_intent_counts WHERE uid=?', ('master',))
conn.commit()
conn.close()
print('Trait evolution counters reset. Traits will stabilize naturally.')
"
```

## Профилактика

1. **Еженедельный бэкап БД**:
   ```bash
   cp memory.db backups/memory_$(date +%Y%m%d).db
   ```

2. **Мониторинг дрейфа личности**:
   Запускать `python tests/run_scenarios.py` раз в неделю для проверки соответствия _SEED.

3. **Логирование инцидентов**:
   Все случаи сбоев записывать в `logs/incidents.log` с датой и описанием.

## Контакты
Если ни одно из решений не помогло -- создать issue с логами из `logs/error.log` и дампом БД.
