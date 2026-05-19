"""Pre-flight warmup. Run this once, well before the talk.

Goal: prime the container image cache in Artifact Registry so the
live demos all hit the cache (~30s startup) instead of triggering a
cold build (~5 min). Each accelerator + requirements set produces a
distinct image hash, so we touch every accelerator the talk uses.

What this DOES warm:
    • CPU image (used by demos 03-08)
    • TPU v5litepod-1 image (used by demo 02)

What this does NOT warm (on purpose):
    • Data() upload caches. Demo 03's punchline is "first run uploads,
      second run is a cache hit" — pre-warming the data would kill it.

Run from the same working directory as the rest of the demo scripts
(so requirements.txt / pyproject.toml hashing matches):
    python 00_warmup.py

Re-run any time the kinetic version, requirements, or base image
changes — the cache key flips and a fresh build is needed.
"""

import time

import kinetic


@kinetic.run(accelerator="tpu-v5litepod-1")
def warm_tpu_image():
  import jax
  import keras

  return {
    "image": "tpu-v5litepod-1",
    "jax": jax.__version__,
    "keras": keras.__version__,
  }


@kinetic.run(accelerator="cpu")
def warm_cpu_image():
  import jax
  import keras
  import orbax.checkpoint as ocp

  return {
    "image": "cpu",
    "jax": jax.__version__,
    "keras": keras.__version__,
    "orbax": ocp.__version__,
  }


def _time(label, fn):
  start = time.time()
  result = fn()
  elapsed = time.time() - start
  print(f"  {label:>30s}  ✓  {elapsed:6.1f}s   {result}")


if __name__ == "__main__":
  print("Warming container images for the demo cluster...\n")
  print(
    "(First run may take ~5 min per image; subsequent runs are cache hits.)\n"
  )
  _time("tpu-v5litepod-1", warm_tpu_image)
  _time("cpu", warm_cpu_image)
  print("\nAll images warm. Live demos should now start in ~30s each.")
