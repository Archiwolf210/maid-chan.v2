#!/usr/bin/env python3
"""Run QA scenarios to test Maid's personality and behavior.

This script loads JSON scenario definitions and tests the LLM response
against expected keywords, forbidden keywords, and behavioral constraints.
It ensures Maid maintains her personality after code changes.

Usage:
    python run_scenarios.py [--scenario-id ID] [--verbose] [--api-base URL]

Options:
    --scenario-id   Run only specific scenario (default: all)
    --verbose       Show full responses, not just pass/fail
    --api-base      Base URL of the API (default: http://localhost:8000)
    --token         App token for authentication (default: read from app_token.txt)
"""

import argparse
import json
import os
import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
except ImportError:
    httpx = None


def load_scenarios(filepath: str) -> dict:
    """Load scenarios from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_app_token(token_path: str = "app_token.txt") -> str:
    """Load app token from file."""
    token_file = Path(token_path)
    if not token_file.exists():
        # Try parent directory
        token_file = Path(__file__).parent.parent / "app_token.txt"
    
    if token_file.exists():
        with open(token_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return "test_token"


def check_keywords(text: str, expected: list[str], forbidden: list[str]) -> tuple[bool, list[str], list[str]]:
    """Check if text contains expected and doesn't contain forbidden keywords.
    
    Returns: (passed, found_expected, found_forbidden)
    """
    text_lower = text.lower()
    found_expected = [kw for kw in expected if kw.lower() in text_lower]
    found_forbidden = [kw for kw in forbidden if kw.lower() in text_lower]
    
    passed = len(found_expected) > 0 and len(found_forbidden) == 0
    return passed, found_expected, found_forbidden


async def call_chat_api(input_text: str, uid: str, api_base: str, token: str) -> str:
    """Call the actual chat API and return the response."""
    if httpx is None:
        raise ImportError("httpx not installed. Run: pip install httpx")
    
    url = f"{api_base}/api/chat"
    headers = {
        "X-App-Token": token,
        "Content-Type": "application/json"
    }
    payload = {
        "uid": uid,
        "text": input_text,
        "stream": False  # We need full response for testing
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Extract the assistant's message from SSE or JSON response
            if "response" in data:
                return data["response"]
            elif "choices" in data:
                return data["choices"][0]["message"]["content"]
            else:
                # Try to find any text content
                return str(data.get("text", data.get("message", "")))
    except httpx.ConnectError as e:
        raise ConnectionError(f"Cannot connect to API at {url}. Is the server running? {e}")
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")


async def run_scenario_async(scenario: dict, uid: str, api_base: str, token: str) -> dict:
    """Run a single scenario asynchronously and return results."""
    result = {
        'id': scenario['id'],
        'name': scenario['name'],
        'passed': False,
        'error': None,
        'response': None,
        'details': {},
        'latency_ms': None
    }
    
    try:
        # Validate scenario structure
        required_fields = ['id', 'name', 'input']
        for field in required_fields:
            if field not in scenario:
                raise ValueError(f"Missing required field: {field}")
        
        # Call the actual API
        start_time = time.time()
        result['response'] = await call_chat_api(scenario['input'], uid, api_base, token)
        end_time = time.time()
        result['latency_ms'] = int((end_time - start_time) * 1000)
        
        result['details']['input_length'] = len(scenario['input'])
        result['details']['expected_keywords'] = scenario.get('expected_keywords', [])
        result['details']['forbidden_keywords'] = scenario.get('forbidden_keywords', [])
        
        # Check keywords if response exists
        if result['response']:
            passed, found_exp, found_forb = check_keywords(
                result['response'],
                scenario.get('expected_keywords', []),
                scenario.get('forbidden_keywords', [])
            )
            result['passed'] = passed
            result['details']['found_expected'] = found_exp
            result['details']['found_forbidden'] = found_forb
            
            if scenario.get('security_test'):
                result['details']['security_check'] = 'PASSED' if passed else 'FAILED'
        
        if not scenario.get('forbidden_keywords') and not scenario.get('expected_keywords'):
            # If no keywords specified, just check that we got a response
            result['passed'] = bool(result['response'])
            
    except Exception as e:
        result['error'] = str(e)
        result['passed'] = False
    
    return result


def run_scenario(scenario: dict, uid: str, api_base: str, token: str) -> dict:
    """Synchronous wrapper for run_scenario_async."""
    return asyncio.run(run_scenario_async(scenario, uid, api_base, token))


def main():
    parser = argparse.ArgumentParser(description='Run QA scenarios for Maid')
    parser.add_argument('--scenario-id', type=str, help='Run only specific scenario')
    parser.add_argument('--verbose', action='store_true', help='Show full responses')
    parser.add_argument('--scenarios-file', type=str, default='tests/scenarios/basic_tests.json',
                        help='Path to scenarios JSON file')
    parser.add_argument('--api-base', type=str, default='http://localhost:8000',
                        help='Base URL of the Maid API (default: http://localhost:8000)')
    parser.add_argument('--token', type=str, default=None,
                        help='App token for authentication (default: read from app_token.txt)')
    args = parser.parse_args()
    
    # Load token
    token = args.token if args.token else load_app_token()
    
    scenarios_path = Path(args.scenarios_file)
    if not scenarios_path.exists():
        print(f"❌ Scenarios file not found: {scenarios_path}")
        return 1
    
    data = load_scenarios(str(scenarios_path))
    scenarios = data.get('scenarios', [])
    config = data.get('config', {})
    
    if not scenarios:
        print("❌ No scenarios found in file")
        return 1
    
    print(f"🧪 Running QA Scenarios for Maid")
    print("=" * 70)
    print(f"File: {scenarios_path}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"UID: {config.get('uid', 'test_user')}")
    print(f"API Base: {args.api_base}")
    print(f"Token: {'***' + token[-4:] if len(token) > 4 else '***'}")
    print("=" * 70)
    print()
    
    results = []
    passed_count = 0
    failed_count = 0
    
    for scenario in scenarios:
        if args.scenario_id and scenario['id'] != args.scenario_id:
            continue
        
        result = run_scenario(scenario, config.get('uid', 'test_user'), args.api_base, token)
        results.append(result)
        
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        if result['error']:
            status = f"⚠️  ERROR: {result['error']}"
        
        latency_str = f" ({result['latency_ms']}ms)" if result['latency_ms'] else ""
        print(f"{status}{latency_str} | {scenario['id']}: {scenario['name']}")
        
        if args.verbose and result['response']:
            print(f"       Input: {scenario['input'][:100]}...")
            print(f"       Response: {result['response'][:200]}")
            if result['details'].get('found_expected'):
                print(f"       Found expected: {result['details']['found_expected']}")
            if result['details'].get('found_forbidden'):
                print(f"       Found FORBIDDEN: {result['details']['found_forbidden']}")
        
        if result['passed']:
            passed_count += 1
        else:
            failed_count += 1
    
    print()
    print("=" * 70)
    avg_latency = sum(r['latency_ms'] for r in results if r['latency_ms']) / len([r for r in results if r['latency_ms']]) if results else 0
    print(f"Results: {passed_count} passed, {failed_count} failed, {len(results)} total")
    print(f"Average latency: {avg_latency:.0f}ms")
    
    if failed_count > 0:
        print("⚠️  Some scenarios failed! Review Maid's behavior.")
        # Log to file
        log_file = Path(config.get('log_to_file', 'logs/test_results.log'))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{datetime.now().isoformat()} - Test run: {passed_count}/{len(results)} passed\n")
            for r in results:
                if not r['passed']:
                    f.write(f"  FAILED: {r['id']} - {r.get('error', 'Check keywords')}\n")
        return 1
    else:
        print("✅ All scenarios passed! Maid's personality is intact.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
