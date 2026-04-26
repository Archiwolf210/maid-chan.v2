#!/usr/bin/env python3
"""Check for memory leaks in Maid application.

This script simulates load by making multiple requests and measuring
memory usage before and after. It helps detect memory leaks in caches,
database connections, or LLM response handling.

Usage:
    python check_memory.py [--requests N] [--uid UID]

Options:
    --requests  Number of simulated requests (default: 100)
    --uid       User ID to use for simulation (default: test_user)
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not installed. Install with: pip install psutil")


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    else:
        # Fallback: try to read from /proc on Linux
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024  # kB to MB
        except:
            pass
        return 0.0


def simulate_chat_request(uid: str = "test_user") -> bool:
    """Simulate a single chat request.
    
    This is a lightweight simulation that exercises the main code paths:
    - Loading state
    - Building cognitive frame
    - LTM retrieval
    - Prompt building
    
    Returns True if successful, False if error.
    """
    try:
        # Import main modules to exercise code paths
        from app.db import db
        from app.memory import get_ltm_relevant, load_rp_scene
        
        # Simulate loading state
        with db() as c:
            c.execute("SELECT 1")
        
        # Simulate LTM retrieval
        get_ltm_relevant(uid, "тестовый запрос для проверки памяти", limit=5)
        
        # Simulate RP scene load
        load_rp_scene(uid)
        
        return True
    except Exception as e:
        print(f"Error in simulation: {e}")
        return False


def run_memory_test(num_requests: int = 100, uid: str = "test_user") -> dict:
    """Run memory leak test.
    
    Returns dict with before/after memory usage and statistics.
    """
    print(f"🧪 Running memory leak test with {num_requests} requests...")
    print("=" * 60)
    
    # Force garbage collection before test
    gc.collect()
    time.sleep(0.5)
    
    mem_before = get_memory_usage_mb()
    print(f"Memory before: {mem_before:.2f} MB")
    
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for i in range(num_requests):
        if simulate_chat_request(uid):
            success_count += 1
        else:
            fail_count += 1
        
        # Progress indicator every 20 requests
        if (i + 1) % 20 == 0:
            current_mem = get_memory_usage_mb()
            print(f"  Request {i+1}/{num_requests} - Memory: {current_mem:.2f} MB")
    
    elapsed = time.time() - start_time
    
    # Force garbage collection after test
    gc.collect()
    time.sleep(0.5)
    
    mem_after = get_memory_usage_mb()
    print(f"Memory after:  {mem_after:.2f} MB")
    
    mem_delta = mem_after - mem_before
    mem_per_request = mem_delta / num_requests if num_requests > 0 else 0
    
    results = {
        'mem_before_mb': mem_before,
        'mem_after_mb': mem_after,
        'mem_delta_mb': mem_delta,
        'mem_per_request_kb': mem_per_request * 1024,
        'success_count': success_count,
        'fail_count': fail_count,
        'elapsed_seconds': elapsed,
        'requests_per_second': num_requests / elapsed if elapsed > 0 else 0
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Check for memory leaks in Maid')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests to simulate')
    parser.add_argument('--uid', type=str, default='test_user', help='User ID for simulation')
    args = parser.parse_args()
    
    if not HAS_PSUTIL:
        print("⚠️  Running without psutil. Results may be less accurate.")
        print("   Install psutil for better memory tracking: pip install psutil")
        print()
    
    results = run_memory_test(args.requests, args.uid)
    
    print()
    print("=" * 60)
    print("📊 RESULTS:")
    print(f"  Requests:      {results['success_count']} succeeded, {results['fail_count']} failed")
    print(f"  Time elapsed:  {results['elapsed_seconds']:.2f}s ({results['requests_per_second']:.1f} req/s)")
    print(f"  Memory delta:  {results['mem_delta_mb']:+.2f} MB total")
    print(f"  Per request:   {results['mem_per_request_kb']:+.2f} KB")
    print()
    
    # Heuristics for detecting leaks
    if results['mem_delta_mb'] > 50:
        print("❌ WARNING: Large memory increase detected (>50MB). Possible leak!")
        print("   Check: caches, database connections, LLM response buffering")
        return 1
    elif results['mem_delta_mb'] > 10:
        print("⚠️  CAUTION: Moderate memory increase (>10MB). Monitor over time.")
        return 0
    else:
        print("✅ Memory usage looks stable. No obvious leaks detected.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
