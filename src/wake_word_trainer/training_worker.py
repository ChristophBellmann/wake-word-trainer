"""Initialize optional random seeds in the fresh training process."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--runtime")
    args, remaining = parser.parse_known_args()
    if args.runtime:
        import yaml

        from .gpu import lease, tensorflow_device

        config = yaml.safe_load(Path(args.runtime).read_text())
        selected = config.get("device", "auto")
        hidden = any(
            os.environ.get(key) == "-1"
            for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        )
        guard = lease("student-training") if selected != "cpu" and not hidden else contextlib.nullcontext()
        with guard:
            runtime = tensorflow_device(selected, int(config.get("compute_threads", 4)))
            Path(args.runtime).parent.joinpath("training_runtime.json").write_text(json.dumps(runtime, indent=2))
            print(f"Training device: {runtime['device']} {runtime['gpu']}", flush=True)
            if config.get("engine") == "optimized":
                import microwakeword.train as backend

                from .optimized_tf import install

                install(backend)
            execute(args, remaining)
    else:
        execute(args, remaining)


def execute(args, remaining):
    if args.seed is not None:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(args.seed)
    if args.control:
        import microwakeword.train as backend

        from .training_control import install

        install(backend)
    sys.argv = ["microwakeword.model_train_eval", *remaining]
    runpy.run_module("microwakeword.model_train_eval", run_name="__main__")


if __name__ == "__main__":
    main()
