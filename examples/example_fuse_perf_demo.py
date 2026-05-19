"""Demo: GCS FUSE vs. download performance for partial-read workloads.

Demonstrates the canonical FUSE-wins scenario: a large dataset where the
user's code only reads a small fraction of files. With the default
download mode the pod must transfer the entire dataset before execution.
With ``fuse=True``, only files actually opened are fetched from GCS.

The script builds a ~2 GB synthetic dataset locally (200 random-byte
shards of 10 MB each), then runs the same remote function twice:

  * Run A: ``Data(dataset_dir)`` (default). Kinetic downloads all 2 GB
    onto the pod's local disk before the function starts.
  * Run B: ``Data(dataset_dir, fuse=True)``. The pod mounts the GCS
    prefix via the GCS FUSE CSI driver; only the 5 files the function
    opens (~50 MB) are fetched from GCS.

Both runs read the same 5 files inside the function and time the read.
The script prints an end-to-end wall-clock comparison plus the in-pod
read time so you can see where FUSE's win comes from: avoiding the
pre-execution bulk download.

Run::

    python examples/example_fuse_perf_demo.py

Requires a GKE cluster with the GCS FUSE CSI driver addon enabled
(``kinetic up`` enables it by default).
"""

import os
import shutil
import tempfile
import time

import kinetic
from kinetic import Data

NUM_FILES = 200
FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per shard
FILES_TO_READ = 5  # ~2.5% of the dataset


def _build_dataset(root: str) -> str:
  """Create NUM_FILES shards of random bytes under root/dataset."""
  dataset_dir = os.path.join(root, "dataset")
  os.makedirs(dataset_dir, exist_ok=True)
  total_mb = (NUM_FILES * FILE_SIZE_BYTES) // (1024 * 1024)
  print(
    f"Building dataset: {NUM_FILES} files x "
    f"{FILE_SIZE_BYTES // (1024 * 1024)} MB = {total_mb} MB total"
  )
  # Random bytes ensure GCS / transport layers can't dedup or compress
  # the payload and skew the comparison.
  for i in range(NUM_FILES):
    with open(os.path.join(dataset_dir, f"shard_{i:04d}.bin"), "wb") as f:
      f.write(os.urandom(FILE_SIZE_BYTES))
  print(f"Dataset built at {dataset_dir}\n")
  return dataset_dir


def make_read_subset(dataset_dir: str, fuse: bool):
  """Build a remote function that reads the first FILES_TO_READ shards.

  The decorator is constructed once per (dataset_dir, fuse) pair because
  the volume spec is bound at decoration time.
  """

  @kinetic.run(
    accelerator="cpu",
    volumes={"/data": Data(dataset_dir, fuse=fuse)},
  )
  def read_subset():
    files_sorted = sorted(os.listdir("/data"))
    targets = files_sorted[:FILES_TO_READ]

    start = time.time()
    bytes_read = 0
    for name in targets:
      with open(f"/data/{name}", "rb") as f:
        bytes_read += len(f.read())
    elapsed = time.time() - start

    return {
      "mode": "fuse" if fuse else "download",
      "files_read": len(targets),
      "files_available": len(files_sorted),
      "bytes_read": bytes_read,
      "user_code_seconds": elapsed,
    }

  return read_subset


def _print_report(
  dl_result: dict,
  dl_e2e: float,
  fuse_result: dict,
  fuse_e2e: float,
) -> None:
  total_bytes = NUM_FILES * FILE_SIZE_BYTES
  total_mb = total_bytes // (1024 * 1024)
  file_mb = FILE_SIZE_BYTES // (1024 * 1024)
  read_pct = 100.0 * dl_result["bytes_read"] / total_bytes
  speedup = dl_e2e / fuse_e2e if fuse_e2e > 0 else float("inf")

  bar = "=" * 60
  print()
  print(bar)
  print("  GCS FUSE vs. Download - partial-read benchmark")
  print(bar)
  print(
    f"Dataset:     {NUM_FILES} files x {file_mb} MB  =  {total_mb} MB total"
  )
  print(
    f"Files read:  {dl_result['files_read']} / "
    f"{dl_result['files_available']}  ({read_pct:.1f}% of bytes)"
  )
  print()
  print(f"{'':28}{'Download':>12}{'FUSE':>12}")
  print(f"{'End-to-end wall-clock':28}{dl_e2e:>11.2f}s{fuse_e2e:>11.2f}s")
  print(
    f"{'User-code read time':28}"
    f"{dl_result['user_code_seconds']:>11.2f}s"
    f"{fuse_result['user_code_seconds']:>11.2f}s"
  )
  print()
  if speedup >= 1.0:
    print(f"FUSE was {speedup:.1f}x faster end-to-end.")
  else:
    print(
      f"FUSE was {1.0 / speedup:.1f}x SLOWER end-to-end. "
      f"Unexpected for a partial-read workload - check cluster state, "
      f"cold-start variance, or whether the dataset is large enough "
      f"for the download cost to dominate."
    )
  print(bar)


def main():
  tmp_root = tempfile.mkdtemp(prefix="kn-fuse-perf-demo-")
  print(f"Temp root: {tmp_root}\n")
  try:
    dataset_dir = _build_dataset(tmp_root)

    # Run A: default download mode. Pod downloads the full 2 GB before
    # user code starts.
    print("Run A: default download mode (fuse=False)")
    download_fn = make_read_subset(dataset_dir, fuse=False)
    t0 = time.time()
    dl_result = download_fn()
    dl_e2e = time.time() - t0
    print(
      f"  end-to-end: {dl_e2e:.2f}s  |  inner read: "
      f"{dl_result['user_code_seconds']:.2f}s\n"
    )

    # Run B: FUSE mode. The same dataset is already cached in GCS from
    # Run A (content-addressed upload cache), so the local upload is a
    # no-op and the only thing we're measuring is pod-side behaviour.
    print("Run B: FUSE mode (fuse=True)")
    fuse_fn = make_read_subset(dataset_dir, fuse=True)
    t0 = time.time()
    fuse_result = fuse_fn()
    fuse_e2e = time.time() - t0
    print(
      f"  end-to-end: {fuse_e2e:.2f}s  |  inner read: "
      f"{fuse_result['user_code_seconds']:.2f}s"
    )

    _print_report(dl_result, dl_e2e, fuse_result, fuse_e2e)
  finally:
    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
  main()
