"""Allows `python -m deploy ...`."""
import sys

from deploy.cli import main

sys.exit(main())
