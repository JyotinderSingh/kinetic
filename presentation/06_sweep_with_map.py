"""Demo 4 — Hyperparameter sweep with kinetic.map().

One function. A list of configs. kinetic.map() fans out across the
cluster, gives you a single BatchHandle, and handles ordering,
streaming, failures, and cleanup.

Run:
    python 07_sweep_with_map.py

Talk track:
    "The function never changed. I'm passing three valid learning rates,
     plus one deliberately broken one to show fault tolerance.
     Notice three things: (1) jobs run in parallel up to max_concurrent=3,
     (2) as_completed yields jobs in finish order, not input order,
     (3) the bad config fails on its own — it doesn't take down the batch."
"""

import os
import time

os.environ["KERAS_BACKEND"] = "jax"

import kinetic


@kinetic.run(accelerator="cpu")
def train(lr: float, epochs: int):
  import keras
  import numpy as np

  if lr > 1.0:
    raise ValueError(
      f"Learning rate {lr} exceeds stability threshold — aborting."
    )

  model = keras.Sequential(
    [
      keras.layers.Dense(64, activation="relu", input_shape=(10,)),
      keras.layers.Dense(1),
    ]
  )
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

  x = np.random.randn(800, 10)
  y = np.random.randn(800, 1)
  history = model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
  return {
    "lr": lr,
    "epochs": epochs,
    "loss": float(history.history["loss"][-1]),
  }


def main():
  # 4 configs: 3 valid, 1 failing to demonstrate graceful failure isolation
  configs = [
    {"lr": 1e-4, "epochs": 4},
    {"lr": 1e-3, "epochs": 4},
    {"lr": 1e-2, "epochs": 4},
    {"lr": 1e10, "epochs": 4},  # deliberately broken
  ]

  print(f"Launching sweep over {len(configs)} configs (max 3 concurrent)...")
  t0 = time.time()
  batch = kinetic.run_async_map(train, configs, max_concurrent=3)
  print(f"Batch group ID: {batch.group_id}\n")

  print("--- Streaming completions (as_completed) ---")
  for job in batch.as_completed(timeout=900):
    try:
      r = job.result(cleanup=False)
      print(f"  {job.job_id}  ✓  lr={r['lr']:<8}  loss={r['loss']:.4f}")
    except Exception as e:
      print(f"  {job.job_id}  ✗  Error: {e}")

  print("\n--- Final tally ---")
  print(f"  status counts: {batch.status_counts()}")
  print(f"  failed jobs:   {len(batch.failures())}")

  results = batch.results(return_exceptions=True, cleanup=False)
  ok = [r for r in results if not isinstance(r, Exception)]
  if ok:
    best = min(ok, key=lambda r: r["loss"])
    print(f"  best config:   lr={best['lr']}, loss={best['loss']:.4f}")

  batch.cleanup()
  print(
    f"\nBatch cleaned up successfully. [Total sweep elapsed: {time.time() - t0:.2f}s]"
  )


if __name__ == "__main__":
  main()
