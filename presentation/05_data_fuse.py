"""Demo 2b — FUSE mounts: lazy-load huge data, never download it.

Same Data() API, one extra flag: `fuse=True`. Kinetic uploads to GCS
once, then mounts the bucket inside the pod via the GCS FUSE CSI
driver. Files are fetched on open() — never as a bulk download.

Use this when:
  - The dataset is much bigger than the pod's disk.
  - You only read a subset of files per run.
  - You're sharing one dataset across many concurrent jobs.

Run:
    python 04_data_fuse.py

Talk track:
    "Same Data() wrapper, one new flag: fuse=True. Imagine /data is
     a 1TB ImageNet shard — the pod doesn't even have that much disk.
     With FUSE, we mount it. Files only stream over the wire when
     the code actually opens them. The function code is identical."
"""

import os
import shutil
import tempfile
import time

import kinetic
from kinetic import Data


@kinetic.run(accelerator="cpu")
def train_with_fuse(dataset_dir: str, target_shards: list[int]):
  import os
  import numpy as np

  all_shards = sorted(os.listdir(dataset_dir))

  # Process ONLY the requested subset of shards (streaming on-demand)
  total_samples = 0
  for shard_idx in target_shards:
    # File contents stream dynamically over FUSE upon calling loadtxt
    filepath = f"{dataset_dir}/shard_{shard_idx:04d}.csv"
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    total_samples += len(data)

  return {
    "mount_path": dataset_dir,
    "shards_visible": len(all_shards),
    "shards_streamed": len(target_shards),
    "samples_processed": total_samples,
    "transport": "CSI FUSE lazy mount",
  }


def create_mock_shards():
  tmp = tempfile.mkdtemp(prefix="kn-fuse-demo-")
  dataset_dir = os.path.join(tmp, "imagenet_shards")
  os.makedirs(dataset_dir, exist_ok=True)

  # Generate 100 mock data shards
  for i in range(100):
    with open(os.path.join(dataset_dir, f"shard_{i:04d}.csv"), "w") as f:
      f.write("embedding_dim,value\n" + "0.512,1.24\n" * 50)
  return tmp, dataset_dir


def main():
  tmp, dataset_dir = create_mock_shards()

  print(f"Local dataset: {dataset_dir} (100 synthetic shards generated)\n")
  print("Mounting via FUSE — no bulk download, files stream on open():")

  try:
    t0 = time.time()

    # Dispatch job requesting exactly 3 specific shards
    selected = [5, 42, 89]
    print(f"Dispatching task targeting shard indices: {selected}")
    result = train_with_fuse(Data(dataset_dir, fuse=True), selected)
    print(f"  → {result}")
    print(f"  [Elapsed: {time.time() - t0:.2f}s]")
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
  main()
