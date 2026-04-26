"""Small, dependency-light helpers for the rest of the package.

Currently:
  - style_filter.extract_json_safely: robust JSON parser that survives the
    common malformations local LLMs (Qwen3, Llama-3, etc.) emit.
"""
