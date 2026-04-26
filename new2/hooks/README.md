# Hooks System for Maid

This directory contains optional hooks that extend Maid's functionality without modifying core code.

## How It Works

If a hook file exists in this directory, it will be automatically called at specific events.

## Available Hooks

### `hooks/on_diary.py`
Called when Maid writes a diary entry.

**Signature:**
```python
def on_diary(uid: str, entry: str, metadata: dict) -> None:
    """Called after Maid writes a diary entry.
    
    Args:
        uid: User ID
        entry: Diary text
        metadata: Dict with humanity_level, voice_band, trigger_reason
    """
    pass
```

**Example usage:**
```python
# Send Telegram notification when diary is written
import requests

def on_diary(uid: str, entry: str, metadata: dict):
    if metadata.get('trigger_reason') == 'emotional_peak':
        # Send to Telegram
        TELEGRAM_TOKEN = "your_token"
        CHAT_ID = "your_chat_id"
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": f"📔 Maid wrote: {entry[:100]}..."}
        )
```

### `hooks/on_letter.py`
Called when Maid composes a letter.

**Signature:**
```python
def on_letter(uid: str, letter: dict) -> None:
    """Called after Maid composes a letter.
    
    Args:
        uid: User ID
        letter: Dict with subject, body, trigger, emotion
    """
    pass
```

### `hooks/on_reflection.py`
Called when Maid completes a self-reflection session.

**Signature:**
```python
def on_reflection(uid: str, insights: list[str]) -> None:
    """Called after Maid completes self-reflection.
    
    Args:
        uid: User ID
        insights: List of insight strings
    """
    pass
```

### `hooks/on_state_change.py`
Called when Maid's emotional state changes significantly.

**Signature:**
```python
def on_state_change(uid: str, old_state: dict, new_state: dict) -> None:
    """Called when Maid's state changes.
    
    Args:
        uid: User ID
        old_state: Previous state dict (mood, fear, attachment, etc.)
        new_state: New state dict
    """
    pass
```

## Enabling Hooks

Simply create the hook file in this directory. The system will automatically detect and call it.

To disable a hook, rename it to `on_diary.py.disabled` or delete it.

## Error Handling

If a hook raises an exception, it will be logged but won't crash the main application.

## Security Notes

- Hooks run with the same permissions as the main application
- Don't store secrets in hook files - use environment variables
- Validate any external API responses before using them
