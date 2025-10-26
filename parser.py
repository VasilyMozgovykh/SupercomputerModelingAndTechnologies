import os
import pandas as pd


def parse_elapsed_time(path):
    elapsed_time = None
    with open(path, "r") as fp:
        for line in fp:
            if "Elapsed time:" in line:
                elapsed_time = float(line.split("Elapsed time: ")[1].split(" seconds")[0])
    return elapsed_time


def parse_iter_num(path):
    iter_num = 0
    with open(path, "r") as fp:
        for line in fp:
            if line.startswith("Step "):
                iter_num = int(line.split("Step ")[1].split(":")[0]) + 1
    return iter_num


if __name__ == "__main__":
    results = []

    cpu_prefix = "bsub_results/cpu"
    for path in os.listdir(cpu_prefix):
        full_path = os.path.join(cpu_prefix, path)
        grid_size = path.split("_")[1].split(".txt")[0]
        if path.startswith("fout"):
            elapsed_time = parse_elapsed_time(full_path)
            iter_num = parse_iter_num(full_path)
            results.append(
                {
                    "kind": "cpu",
                    "grid_size": grid_size,
                    "iter_num": iter_num,
                    "elapsed_time": elapsed_time,
                    "num_threads": 1,
                    "xdim": int(grid_size.split("x")[0]),
                    "speedup": 1.,
                }
            )
    cpu_df = pd.DataFrame(results)

    omp_prefix = "bsub_results/omp"
    for path in os.listdir(omp_prefix):
        full_path = os.path.join(omp_prefix, path)
        grid_size = path.split("_")[1]
        num_threads = path.split("_")[2].split(".txt")[0]
        if path.startswith("fout"):
            elapsed_time = parse_elapsed_time(full_path)
            iter_num = parse_iter_num(full_path)
            results.append(
                {
                    "kind": f"omp_{num_threads}",
                    "grid_size": grid_size,
                    "iter_num": iter_num,
                    "elapsed_time": elapsed_time,
                    "num_threads": int(num_threads),
                    "xdim": int(grid_size.split("x")[0]),
                    "speedup": float(cpu_df.loc[cpu_df.grid_size==grid_size, "elapsed_time"].to_numpy()[0]) / elapsed_time,
                }
            )

    (
        pd.DataFrame(results)
        .sort_values(by=["xdim", "num_threads"])
        .drop(columns=["num_threads", "xdim"])
        .to_csv("results_table.txt", sep="&", index=False)
    )