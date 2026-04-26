#!/usr/bin/env python3
"""Run QA scenarios to test Maid's personality and behavior.

This script loads JSON scenario definitions and tests the LLM response
against expected keywords, forbidden keywords, and behavioral constraints.
It ensures Maid maintains her personality after code changes.

Usage:
    python run_scenarios.py [--scenario-id ID] [--verbose]

Options:
    --scenario-id   Run only specific scenario (default: all)
    --verbose       Show full responses, not just pass/fail
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_scenarios(filepath: str) -> dict:
    """Load scenarios from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_keywords(text: str, expected: list[str], forbidden: list[str]) -> tuple[bool, list[str], list[str]]:
    """Check if text contains expected and doesn't contain forbidden keywords.
    
    Returns: (passed, found_expected, found_forbidden)
    """
    text_lower = text.lower()
    found_expected = [kw for kw in expected if kw.lower() in text_lower]
    found_forbidden = [kw for kw in forbidden if kw.lower() in text_lower]
    
    passed = len(found_expected) > 0 and len(found_forbidden) == 0
    return passed, found_expected, found_forbidden


def run_scenario(scenario: dict, uid: str = "test_user") -> dict:
    """Run a single scenario and return results.
    
    This is a SIMULATED test runner. In real usage, it would call the actual API.
    For now, it validates scenario structure and provides a framework.
    """
    result = {
        'id': scenario['id'],
        'name': scenario['name'],
        'passed': False,
        'error': None,
        'response': None,
        'details': {}
    }
    
    try:
        # Validate scenario structure
        required_fields = ['id', 'name', 'input']
        for field in required_fields:
            if field not in scenario:
                raise ValueError(f"Missing required field: {field}")
        
        # In a real implementation, this would call the chat API:
        # response = await call_chat_api(scenario['input'], uid)
        # result['response'] = response
        
        # For now, we simulate with a placeholder
        result['response'] = "[SIMULATED RESPONSE - Implement API call in run_scenarios.py]"
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
        
        if not scenario.get('forbidden_keywords'):
            # If no forbidden keywords, just check for expected
            result['passed'] = len(result['details'].get('found_expected', [])) > 0
        
    except Exception as e:
        result['error'] = str(e)
        result['passed'] = False
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Run QA scenarios for Maid')
    parser.add_argument('--scenario-id', type=str, help='Run only specific scenario')
    parser.add_argument('--verbose', action='store_true', help='Show full responses')
    parser.add_argument('--scenarios-file', type=str, default='tests/scenarios/basic_tests.json',
                        help='Path to scenarios JSON file')
    args = parser.parse_args()
    
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
    print("=" * 70)
    print()
    
    results = []
    passed_count = 0
    failed_count = 0
    
    for scenario in scenarios:
        if args.scenario_id and scenario['id'] != args.scenario_id:
            continue
        
        result = run_scenario(scenario, config.get('uid', 'test_user'))
        results.append(result)
        
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        if result['error']:
            status = f"⚠️  ERROR: {result['error']}"
        
        print(f"{status} | {scenario['id']}: {scenario['name']}")
        
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
    print(f"Results: {passed_count} passed, {failed_count} failed, {len(results)} total")
    
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
