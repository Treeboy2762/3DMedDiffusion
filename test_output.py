#!/usr/bin/env python3
"""
Test script to verify SLURM output works
"""

import sys
import os
import time

# Force unbuffered output for SLURM
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("="*60)
print("TESTING SLURM OUTPUT")
print("="*60)
sys.stdout.flush()

print("Step 1: Starting test...")
sys.stdout.flush()
time.sleep(1)

print("Step 2: Loading models...")
sys.stdout.flush()
time.sleep(2)

print("Step 3: Processing...")
sys.stdout.flush()
time.sleep(1)

print("Step 4: Complete!")
print("This should appear in your SLURM .out file")
sys.stdout.flush()

print("\n✅ SLURM output test completed successfully!")
sys.stdout.flush()



