#!/usr/bin/env python
"""
Test script to verify D2DS bundle exemption fix.
Verifies that previous week's DH conditional bundles can be reused for D2DS.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.bundle_tracker import filter_available_bundles

def test_exemption_mechanism():
    """Test that exemption mechanism works correctly"""
    print("="*70)
    print("Testing Bundle Exemption Mechanism")
    print("="*70)

    # Simulate bundle pool
    all_bundles = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    # Simulate used tracker (bundles 1, 2, 3 were used before)
    used_tracker = {
        'all': {1, 2, 3},
        'dh': {1, 2},
        'd2ds': {3},
        'by_date': {}
    }

    # Test 1: Normal filtering (no exemptions)
    print("\nTest 1: Normal filtering (no exemptions)")
    available = filter_available_bundles(all_bundles, used_tracker)
    expected = {4, 5, 6, 7, 8, 9, 10}
    print(f"  Available bundles: {sorted(available)}")
    print(f"  Expected: {sorted(expected)}")
    assert available == expected, f"Test 1 failed: {available} != {expected}"
    print("  ✓ PASSED")

    # Test 2: With exemptions (bundle 2 should be allowed despite being used)
    print("\nTest 2: With exemptions (bundle 2 exempt)")
    exempt_bundles = {2}
    available_with_exempt = filter_available_bundles(all_bundles, used_tracker, exempt_bundles=exempt_bundles)
    expected_with_exempt = {2, 4, 5, 6, 7, 8, 9, 10}
    print(f"  Available bundles: {sorted(available_with_exempt)}")
    print(f"  Expected: {sorted(expected_with_exempt)}")
    assert available_with_exempt == expected_with_exempt, f"Test 2 failed: {available_with_exempt} != {expected_with_exempt}"
    print("  ✓ PASSED")

    # Test 3: Multiple exemptions
    print("\nTest 3: Multiple exemptions (bundles 1, 2, 3 exempt)")
    exempt_bundles_multi = {1, 2, 3}
    available_multi_exempt = filter_available_bundles(all_bundles, used_tracker, exempt_bundles=exempt_bundles_multi)
    expected_multi_exempt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    print(f"  Available bundles: {sorted(available_multi_exempt)}")
    print(f"  Expected: {sorted(expected_multi_exempt)}")
    assert available_multi_exempt == expected_multi_exempt, f"Test 3 failed: {available_multi_exempt} != {expected_multi_exempt}"
    print("  ✓ PASSED")

    # Test 4: Exempt bundle not in used set (should still work)
    print("\nTest 4: Exempt bundle not previously used (bundle 5)")
    exempt_not_used = {5}
    available_not_used = filter_available_bundles(all_bundles, used_tracker, exempt_bundles=exempt_not_used)
    expected_not_used = {4, 5, 6, 7, 8, 9, 10}  # Same as no exemption since 5 wasn't used
    print(f"  Available bundles: {sorted(available_not_used)}")
    print(f"  Expected: {sorted(expected_not_used)}")
    assert available_not_used == expected_not_used, f"Test 4 failed: {available_not_used} != {expected_not_used}"
    print("  ✓ PASSED")

    print("\n" + "="*70)
    print("All tests PASSED! ✓")
    print("="*70)
    print("\nSummary:")
    print("  The exemption mechanism correctly allows specified bundles")
    print("  to be reused despite the without-replacement rule.")
    print("  This enables D2DS to reuse previous week's DH conditional bundles.")
    print("="*70)

if __name__ == "__main__":
    test_exemption_mechanism()
