from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path


def run_module(module_name):
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    module = import_module(module_name)
    module.run(data_root=args.data_root, output_root=args.output_root)
