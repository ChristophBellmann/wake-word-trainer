"""Initialize optional random seeds in the fresh training process."""

from __future__ import annotations

import argparse
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int)
    args, remaining = parser.parse_known_args()
    if args.seed is not None:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(args.seed)
    sys.argv = ["microwakeword.model_train_eval", *remaining]
    runpy.run_module("microwakeword.model_train_eval", run_name="__main__")


if __name__ == "__main__":
    main()
