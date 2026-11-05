#!/usr/bin/env python3
"""Optional bi-encoder fine-tune. See semsearch.finetune for the implementation.

Offline check, no model download:

    python scripts/finetune.py --dry-run
"""

from semsearch.finetune import main

if __name__ == "__main__":
    raise SystemExit(main())
