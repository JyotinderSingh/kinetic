"""Demo 2a — The Data() API: ship your dataset to the remote pod.

Until now we used keras.datasets, which downloads inside the pod.
Real workloads have data on disk. Wrap it in `Data(...)` and Kinetic
hashes it, uploads it to GCS once, mounts it, and resolves the
argument to a regular filesystem path on the remote.

Run this twice in a row — the second invocation is a cache hit.

Run:
    python 03_data_local.py

Talk track:
    "I wrapped my local path in Data(). On the first run, watch the
     upload. On the second run — same data, no upload. SHA-256 hash
     over the file contents means identical datasets are uploaded
     exactly once. The remote function just sees a plain path string."
"""

import json
import os
import tempfile
import time
import uuid

import kinetic
from kinetic import Data


@kinetic.run(accelerator="cpu")
def train_with_local_data(dataset_dir: str, config_path: str):
  import json
  import numpy as np

  # 1. Load experiment configuration
  with open(config_path) as f:
    config = json.load(f)

  # 2. Read dataset directly from local filesystem path
  train_data = np.loadtxt(f"{dataset_dir}/train.csv", delimiter=",", skiprows=1)
  features, labels = train_data[:, 0], train_data[:, 1]

  # 3. Simulate lightweight training computation
  loss = float(np.mean(labels) * config["lr"])

  return {
    "run_id": config["run_id"],
    "samples_trained": len(features),
    "final_loss": round(loss, 4),
    "status": "completed",
  }


def create_local_dataset():
  tmp = tempfile.mkdtemp(prefix="kn-demo-")
  run_id = str(uuid.uuid4())[:8]

  dataset_dir = os.path.join(tmp, "dataset")
  os.makedirs(dataset_dir, exist_ok=True)
  with open(os.path.join(dataset_dir, "train.csv"), "w") as f:
    f.write("feature,label\n1,100\n2,200\n3,300\n4,400\n5,500\n")
  with open(os.path.join(dataset_dir, "val.csv"), "w") as f:
    f.write(f"# run_id: {run_id}\nfeature,label\n6,600\n7,700\n")

  config_path = os.path.join(tmp, "config.json")
  with open(config_path, "w") as f:
    json.dump({"lr": 0.01, "epochs": 10, "batch_size": 32, "run_id": run_id}, f)
  return tmp, dataset_dir, config_path


def main():
  tmp, dataset_dir, config_path = create_local_dataset()

  print(f"Local artifacts generated at: {tmp}\n")

  print("First call — Expect hashing, artifact packaging, and upload:")
  t0 = time.time()
  result1 = train_with_local_data(Data(dataset_dir), Data(config_path))
  print(f"  → {result1}")
  print(f"  [Elapsed: {time.time() - t0:.2f}s]\n")

  print("Second call — Expect CAS cache hit, instant dispatch (no upload):")
  t1 = time.time()
  result2 = train_with_local_data(Data(dataset_dir), Data(config_path))
  print(f"  → {result2}")
  print(f"  [Elapsed: {time.time() - t1:.2f}s]")


if __name__ == "__main__":
  main()
