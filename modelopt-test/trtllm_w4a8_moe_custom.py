#!/usr/bin/env python3
"""Compatibility CLI for TRT-LLM W4A8_CUSTOM MoE checkpoint prep.

Export now emits TRT-LLM-ready scales via ``modelopt.torch.export.trtllm_w4a8_moe``.
This script remains for rewriting legacy checkpoints exported before that change.
"""

from __future__ import annotations

import sys
from pathlib import Path

from modelopt.torch.export.trtllm_w4a8_moe import rewrite_checkpoint_for_trtllm_w4a8_custom


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Rewrite legacy ModelOpt W4A8_AWQ MoE checkpoint for TRT-LLM CUSTOM_W4A8"
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Rewrite even if marker exists")
    args = parser.parse_args(argv)

    if not rewrite_checkpoint_for_trtllm_w4a8_custom(args.checkpoint_dir, force=args.force):
        print(f"SKIP: not a W4A8_AWQ checkpoint: {args.checkpoint_dir}", file=sys.stderr)
        return 1

    print(f"Prepared {args.checkpoint_dir} for TRT-LLM W4A8_CUSTOM MoE loading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
