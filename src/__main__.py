"""
__main__.py
-----------

Command-line entry point of the **VideoSemanticRepresentation** framework.

This module provides the top-level interface for two main functionalities:

1. **Video Retrieval Mode (default)**
   Compares two videos by their semantic feature representations, and locates
   the most similar segment of the reference video to each query clip.
   Invoked when no explicit subcommand is specified.

   Example:
       python -m src <reference_video> -q <query_video_1> [<query_video_2> ...]

2. **Feature Extraction Mode**
   Extracts and optionally saves semantic feature vectors (.npy) for one or
   multiple videos. This mode supports flexible output handling (single file
   or directory), conflict detection, and interactive overwrite prompts.

   Example:
       python -m src feature <input_video> [-o <output_path>]

Features
--------
- Integrates the feature extraction pipeline (`feature.get_feature`) and
  similarity search module (`search.search`).
- Provides interactive output path resolution via `_resolve_output_paths()`.
- Supports both single-file and batch feature generation.
- Compatible with cached feature reuse to accelerate repeated runs.

Usage
-----
Retrieval mode (default):
    python -m src data/raw/reference.mp4 -q data/slice/query.mp4

Feature extraction mode:
    python -m src feature data/raw/video1.mp4 data/raw/video2.mp4 -o data/features/

Notes
-----
This module is intended to be executed as a package entry point.
Direct imports are possible but not recommended.

Author: Napreth
Version: v0.3.1
"""


import sys
import argparse
from pathlib import Path
from typing import Literal
from .feature import get_feature
from .search import search


src_dir = Path(__file__).resolve().parent
root_dir = src_dir.parent
block = 0.5    # Duration(second) per convolution block


def _resolve_output_paths(inputs: list[str], output: str | None) -> tuple[list[str], Literal["file", "dir"], Path] | None:
    """
    Resolve output saving strategy based on input list and output path.

    This function handles all logic for deciding whether outputs are saved
    as files or directories, including conflict checks and interactive user prompts.

    Parameters
    ----------
    inputs : list of str
        List of input video paths.
    output : str or None
        Target output path provided by user. If None, defaults to current working directory.

    Returns
    -------
    tuple or None
        (filtered_inputs, save_pattern, save_path)
        - filtered_inputs: list[str], remaining inputs after skipping/cancelling
        - save_pattern: 'file' or 'dir'
        - save_path: Path object of resolved output
        Returns None if operation cancelled by user.
    """
    save_pattern: Literal["file", "dir"] = "dir"
    save_path = Path.cwd()

    if not output:
        return inputs, save_pattern, save_path

    original_count = len(inputs)
    output = Path(output).absolute()
    save_path = output

    # --- Case 1: Output is an existing file ---
    if output.is_file():
        if len(inputs) == 1:
            save_pattern = "file"
        else:
            print("[ERROR] Multi-inputs cannot be saved as a file.")
            return None

    # --- Case 2: Output is an existing directory ---
    elif output.is_dir():
        save_pattern = "dir"
        if len(inputs) == 1:
            filename = Path(inputs[0]).name
            target = output / filename
            if target.is_file():
                overwrite = input(f"[WARNING] The file '{target}' already exists. Overwrite? (y/n) ").strip().lower() in ("y", "yes")
                if not overwrite:
                    print("Operation cancelled.")
                    return None
            elif target.is_dir():
                print(f"[ERROR] The directory '{target}' already exists. Please change output path.")
                return None
        else:
            skip_all_files = overwrite_all_files = skip_all_dirs = False
            for idx in range(len(inputs) - 1, -1, -1):
                filename = Path(inputs[idx]).name
                target = output / filename
                # Handle file conflicts
                if target.is_file():
                    if overwrite_all_files:
                        continue
                    if skip_all_files:
                        inputs.pop(idx)
                        continue
                    choice = input(f"[WARNING] The file '{target}' already exists.\n"
                                   "  o: Overwrite\n"
                                   "  s: Skip\n"
                                   "  a: Overwrite All Files\n"
                                   "  k: Skip All Files\n"
                                   "  c: Cancel\n"
                                   "  default = c\n"
                                   ": ").strip().lower()
                    if choice == "o":
                        continue
                    elif choice == "s":
                        inputs.pop(idx)
                    elif choice == "a":
                        overwrite_all_files = True
                        continue
                    elif choice == "k":
                        skip_all_files = True
                        inputs.pop(idx)
                    else:
                        print("Operation cancelled.")
                        return None
                # Handle directory conflicts
                elif target.is_dir():
                    if skip_all_dirs:
                        inputs.pop(idx)
                        continue
                    choice = input(f"[WARNING] The directory '{target}' already exists.\n"
                                   "  s: Skip\n"
                                   "  k: Skip All Directories\n"
                                   "  c: Cancel\n"
                                   "  default = c\n"
                                   ": ").strip().lower()
                    if choice == "s":
                        inputs.pop(idx)
                    elif choice == "k":
                        skip_all_dirs = True
                        inputs.pop(idx)
                    else:
                        print("Operation cancelled.")
                        return None

    # --- Case 3: Output does not exist ---
    else:
        if len(inputs) == 1:
            if output.suffix:
                save_pattern = "file"
            else:
                output.mkdir(parents=True, exist_ok=True)
                save_pattern = "dir"
        else:
            if output.suffix:
                print("[WARNING] Output path looks like a file, treating as directory for multiple inputs.")
            output.mkdir(parents=True, exist_ok=True)
            save_pattern = "dir"

    # --- Post checks ---
    if not inputs:
        print("[WARNING] All inputs were skipped.")
        return None
    elif len(inputs) < original_count:
        print(f"[INFO] {original_count - len(inputs)} input(s) were skipped.")

    return inputs, save_pattern, save_path


def main():
    # 没有参数 → 打印帮助
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python -m src <reference_video> -q <query_video> [<query_video> ...]")
        print("  python -m src feature <input_video> [-o <output_path>]")
        return 0

    # ------------------------------
    #  Feature Extraction Mode
    # ------------------------------
    if sys.argv[1] == "feature":
        parser = argparse.ArgumentParser(
            prog="VideoSemanticRepresentation feature",
            description="Extract and save semantic feature(s) from video(s)."
        )
        parser.add_argument("inputs", nargs="+", help="Input video(s) to process.")
        parser.add_argument("-o", "--output", help="Output path (file or directory).")

        args = parser.parse_args(sys.argv[2:])

        result = _resolve_output_paths(args.inputs, args.output)
        if result is None:
            return 1

        inputs, save_pattern, save_path = result
        for video_path in inputs:
            to_save = str(save_path / Path(video_path).name) if save_pattern == "dir" else str(save_path)
            get_feature(video_path, block, to_save)
        return 0

    # ------------------------------
    #  Retrieval (default) Mode
    # ------------------------------
    parser = argparse.ArgumentParser(
        prog="VideoSemanticRepresentation",
        description="Video semantic retrieval mode (default)."
    )
    parser.add_argument("reference_video", help="Reference video path (used as search base).")
    parser.add_argument(
        "-q", "--query", nargs="+", required=True,
        help="Query video(s) to compare against reference."
    )

    args = parser.parse_args()

    F_path = Path(args.reference_video).resolve()
    F = get_feature(str(F_path), block)
    for q in args.query:
        Q_path = Path(q).resolve()
        Q = get_feature(str(Q_path), block)
        search(block, F, Q, Q_path.name)

    return 0


if __name__ == '__main__':
    sys.exit(main())
