#!/bin/bash
python setup.py build --build-lib build/for-test
PYTHONPATH=build/for-test pytest
