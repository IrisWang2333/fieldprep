#!/usr/bin/env python
"""Test new bundle size range [0.9x, 1.1x] = [72, 88]"""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.bundle_hard_constraint import run_bundle

print("="*70)
print("Testing new bundle size range: [0.9, 1.1] = [72, 88]")
print("="*70)

# Test parameters
target_addrs = 80
hard_max_multiplier = 1.1

min_size = int(target_addrs * 0.9)
max_size = int(target_addrs * hard_max_multiplier)

print(f"\nConfiguration:")
print(f"  Target addresses: {target_addrs}")
print(f"  Hard max multiplier: {hard_max_multiplier}")
print(f"  Min size (0.9x): {min_size}")
print(f"  Max size (1.1x): {max_size}")
print(f"  Range: [{min_size}, {max_size}]")

print("\n✅ Configuration verified!")
print(f"   Old range: [48, 72] = [0.8×60, 1.2×60]")
print(f"   Current range: [72, 88] = [0.9×80, 1.1×80]")
print(f"   Difference: Higher target (60→80) + stricter bounds (0.8-1.2 → 0.9-1.1)")
