"""Build and execute every notebook in a fresh Jupyter kernel."""

import os
import sys
import time
import nbformat as nbf
from nbclient import NotebookClient

import build_nb00
import build_nb01
import build_nb02
import build_nb03
import build_nb04


def execute_notebook(nb: nbf.NotebookNode, working_dir: str = "notebooks") -> nbf.NotebookNode:
    """Execute a notebook exactly as Jupyter does, failing on the first bad cell."""
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": working_dir}},
        allow_errors=False,
        record_timing=False,
    )
    kernel_env = os.environ.copy()
    kernel_env["PATH"] = os.pathsep.join([os.path.dirname(sys.executable), kernel_env.get("PATH", "")])
    kernel_env.setdefault("MPLCONFIGDIR", "/tmp/mlprep-matplotlib")
    kernel_env.setdefault("IPYTHONDIR", "/tmp/mlprep-ipython")
    kernel_env.setdefault("LOKY_MAX_CPU_COUNT", "4")
    return client.execute(env=kernel_env)


def main():
    notebooks_to_build = [
        ("notebooks/00_start_here.ipynb", build_nb00.create_nb00),
        ("notebooks/01_core_concepts.ipynb", build_nb01.create_nb01),
        ("notebooks/02_metrics_and_data.ipynb", build_nb02.create_nb02),
        ("notebooks/03_models_and_training.ipynb", build_nb03.create_nb03),
        ("notebooks/04_review_and_practice.ipynb", build_nb04.create_nb04),
    ]

    total_start = time.time()
    for rel_path, builder in notebooks_to_build:
        print(f"\n========================================================")
        print(f"Building and executing: {rel_path}")
        print(f"========================================================")
        t0 = time.time()
        nb = builder()
        executed_nb = execute_notebook(nb, working_dir="notebooks")

        with open(rel_path, "w", encoding="utf-8") as f:
            nbf.write(executed_nb, f)

        elapsed = time.time() - t0
        print(f"✅ Successfully executed and written {rel_path} in {elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    print(f"\n🎉 All 5 notebooks successfully built and executed in {total_elapsed:.2f}s ({total_elapsed/60:.2f}m)!")


if __name__ == "__main__":
    main()
