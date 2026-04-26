#!/usr/bin/env python3
"""Scan for accidentally committed secrets (API keys, tokens, passwords).

This script checks config files, logs, and source code for patterns
that look like API keys, tokens, or passwords. It should be run before
committing to git or deploying to production.

Usage:
    python scan_secrets.py [--fix]

Options:
    --fix   Rename suspicious files to .bak instead of just warning
"""

import os
import re
import sys
from pathlib import Path

# Patterns that look like secrets
SECRET_PATTERNS = [
    # API Keys
    (r'api[_-]?key\s*[=:]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?', 'API Key'),
    (r'token\s*[=:]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?', 'Token'),
    (r'secret\s*[=:]\s*["\']?([A-Za-z0-9_\-]{16,})["\']?', 'Secret'),
    (r'password\s*[=:]\s*["\']?([^\s"\']{8,})["\']?', 'Password'),
    # AWS
    (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
    (r'aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\']?([A-Za-z0-9/+=]{40})["\']?', 'AWS Secret Key'),
    # OpenAI / Anthropic
    (r'sk-[A-Za-z0-9]{48}', 'OpenAI/API Key'),
    (r'anthropic[_-]?key\s*[=:]\s*["\']?([A-Za-z0-9_\-]{32,})["\']?', 'Anthropic Key'),
    # Generic high-entropy strings
    (r'["\']([A-Za-z0-9+/]{40,}={0,2})["\']', 'High-entropy string (possible key)'),
]

# Files/directories to skip
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env'}
SKIP_FILES = {'scan_secrets.py', 'package-lock.json', 'yarn.lock', 'Pipfile.lock'}


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a single file for secrets. Returns list of (line_no, pattern_name, line)."""
    findings = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_no, line in enumerate(f, 1):
                for pattern, name in SECRET_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip false positives
                        if 'example' in line.lower() or 'your_' in line.lower() or '<your' in line.lower():
                            continue
                        if name == 'Password' and ('password_hash' in line or 'hashed_password' in line):
                            continue
                        findings.append((line_no, name, line.strip()))
    except (PermissionError, UnicodeDecodeError):
        pass
    return findings


def scan_directory(root: Path = None) -> dict[str, list]:
    """Scan directory for secrets. Returns dict of filepath -> findings."""
    if root is None:
        root = Path.cwd()
    
    results = {}
    for filepath in root.rglob('*'):
        if filepath.is_file():
            # Skip directories and binary files
            if any(part in str(filepath) for part in SKIP_DIRS):
                continue
            if filepath.name in SKIP_FILES:
                continue
            if filepath.suffix in {'.pyc', '.pyo', '.so', '.dll', '.exe', '.bin'}:
                continue
            
            findings = check_file(filepath)
            if findings:
                results[str(filepath)] = findings
    
    return results


def main():
    fix_mode = '--fix' in sys.argv
    print("🔍 Scanning for secrets...")
    print("=" * 60)
    
    results = scan_directory()
    
    if not results:
        print("✅ No secrets found! Your code looks clean.")
        return 0
    
    total_findings = sum(len(v) for v in results.values())
    print(f"⚠️  Found {total_findings} potential secret(s) in {len(results)} file(s):\n")
    
    for filepath, findings in results.items():
        print(f"📁 {filepath}")
        for line_no, name, line in findings:
            # Mask the actual secret in output
            masked_line = re.sub(r'[A-Za-z0-9+/=]{20,}', '***REDACTED***', line[:100])
            print(f"   Line {line_no}: {name} -> {masked_line}")
        
        if fix_mode:
            backup_path = filepath.with_suffix(filepath.suffix + '.bak')
            os.rename(filepath, backup_path)
            print(f"   🔒 File renamed to {backup_path.name}")
        print()
    
    print("=" * 60)
    if fix_mode:
        print("🔒 Suspicious files have been renamed to .bak")
        print("   Review them manually and remove secrets before committing.")
    else:
        print("💡 Run with --fix to rename suspicious files to .bak")
        print("   Then review them manually and remove secrets.")
    
    return 1  # Exit with error code to fail CI/CD


if __name__ == '__main__':
    sys.exit(main())
