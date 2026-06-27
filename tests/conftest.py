"""
conftest.py — change cwd to src/python before tests so ctypesroutines
finds libhamiltonian_3rd.so via the relative path './libhamiltonian_3rd.so'.
"""
import os
import sys
import pytest

SRC_PY = os.path.join(os.path.dirname(__file__), "..", "src", "python")
SRC_PY = os.path.abspath(SRC_PY)

def pytest_configure(config):
    os.chdir(SRC_PY)
    if SRC_PY not in sys.path:
        sys.path.insert(0, SRC_PY)
