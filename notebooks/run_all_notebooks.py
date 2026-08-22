"""
Direct, robust in-process notebook builder and executor.
Executes all 5 notebooks, capturing text outputs, tables, and matplotlib figures
directly into standard nbformat v4 cell outputs.
"""

import base64
from contextlib import redirect_stderr, redirect_stdout
import io
import os
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell

import build_nb00
import build_nb01
import build_nb02
import build_nb03
import build_nb04


def execute_notebook_in_process(nb: nbf.NotebookNode, working_dir: str = "notebooks") -> nbf.NotebookNode:
    """Execute all code cells of a notebook sequentially in an in-process IPython session."""
    orig_cwd = os.getcwd()
    os.chdir(working_dir)
    sys.path.insert(0, ".")

    # Initialize fresh interactive shell
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    shell.colors = "NoColor"

    # Set non-interactive matplotlib backend
    matplotlib.use("Agg")

    exec_count = 1

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        source = cell.source
        cell.outputs = []
        cell.execution_count = exec_count

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Clear any existing matplotlib figures
        plt.close("all")

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            res = shell.run_cell(source, store_history=True)

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()

        # 1. Add stdout stream
        if stdout_val:
            cell.outputs.append(
                nbf.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=stdout_val,
                )
            )

        # 2. Add stderr stream (if any)
        if stderr_val:
            cell.outputs.append(
                nbf.v4.new_output(
                    output_type="stream",
                    name="stderr",
                    text=stderr_val,
                )
            )

        # 3. Check for matplotlib plot figures created in cell
        fig_nums = plt.get_fignums()
        for fnum in fig_nums:
            fig = plt.figure(fnum)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=110)
            img_buf.seek(0)
            img_b64 = base64.b64encode(img_buf.read()).decode("utf-8")
            plt.close(fig)

            cell.outputs.append(
                nbf.v4.new_output(
                    output_type="display_data",
                    data={"image/png": img_b64, "text/plain": "<Figure size ...>"},
                )
            )

        # 4. Check for return value / expression display
        if res.result is not None:
            plain_repr = repr(res.result)
            data_dict = {"text/plain": plain_repr}
            if hasattr(res.result, "_repr_html_"):
                data_dict["text/html"] = res.result._repr_html_()
            cell.outputs.append(
                nbf.v4.new_output(
                    output_type="execute_result",
                    execution_count=exec_count,
                    data=data_dict,
                )
            )

        # 5. Check for execution error
        if res.error_in_exec:
            ename = type(res.error_in_exec).__name__
            evalue = str(res.error_in_exec)
            print(f"❌ Cell Error in notebook execution: {ename}: {evalue}")
            print(f"Cell source:\n{source}")
            os.chdir(orig_cwd)
            raise res.error_in_exec

        exec_count += 1

    os.chdir(orig_cwd)
    return nb


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
        executed_nb = execute_notebook_in_process(nb, working_dir="notebooks")

        with open(rel_path, "w", encoding="utf-8") as f:
            nbf.write(executed_nb, f)

        elapsed = time.time() - t0
        print(f"✅ Successfully executed and written {rel_path} in {elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    print(f"\n🎉 All 5 notebooks successfully built and executed in {total_elapsed:.2f}s ({total_elapsed/60:.2f}m)!")


if __name__ == "__main__":
    main()
