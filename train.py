# -*- coding: utf-8 -*-
"""
Thin launcher so that ``python train.py`` works straight from a checkout
(without installing the package first): it puts ``src/`` on ``sys.path``
and delegates to the real driver in ``npltokenizer/train.py``.
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
)

from npltokenizer.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())