"""Explicit local-only validation; no downloads or path discovery."""
import argparse
from ai_research_work.f1.bundle import Bundle
p = argparse.ArgumentParser()
p.add_argument('--bundle')
a = p.parse_args()
if not a.bundle:
    print('SKIP: supply --bundle with a trusted compatible local bundle.')
else:
    try:
        Bundle(a.bundle)
    except (OSError, ValueError, KeyError):
        p.exit(2, 'FAIL: local bundle missing, invalid or incompatible. No download attempted.\n')
    print('PASS: local bundle integrity and compatibility checks.')
