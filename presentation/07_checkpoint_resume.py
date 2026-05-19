"""Demo 5 — Auto-resumable training with Orbax checkpoints.

Every Kinetic job gets a per-job GCS prefix exposed as KINETIC_OUTPUT_DIR.
Write checkpoints there with Orbax. If you rerun the same function,
Orbax discovers the latest step and picks up where it left off.

To show the resume behavior live:
    1. Run once. Let it complete 5 steps.
    2. Run a second time with the SAME job id (use kinetic.attach) —
       or just rerun this script after editing end_step upward.
       It starts at step 5, not 0.

Run:
    python 08_checkpoint_resume.py

Talk track:
    "KINETIC_OUTPUT_DIR is a GCS prefix Kinetic gives every job for
     free. Orbax writes checkpoints there. On a rerun, it sees the
     prior checkpoints and resumes. Your training function doesn't
     care that it's running on a different pod from yesterday's run —
     the state is in GCS, not on the node."
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import kinetic
from kinetic.cli.profiles import resolve_infra
from kinetic.constants import build_bucket_name

infra = resolve_infra()
bucket = build_bucket_name(infra["project"], infra["cluster"])
stable_output_dir = f"gs://{bucket}/outputs/resume_demo"


@kinetic.run(accelerator="cpu", output_dir=stable_output_dir)
def train_with_auto_resume():
  import time

  import jax.numpy as jnp
  import orbax.checkpoint as ocp

  output_dir = os.environ["KINETIC_OUTPUT_DIR"]
  print(f"Checkpoint store: {output_dir}")

  options = ocp.CheckpointManagerOptions(max_to_keep=3)
  mngr = ocp.CheckpointManager(
    output_dir, ocp.StandardCheckpointer(), options=options
  )

  latest = mngr.latest_step()
  if latest is None:
    print("No prior checkpoint — starting from step 0.")
    state = {
      "step": 0,
      "weights": jnp.ones((10, 10)),
      "bias": jnp.zeros((10,)),
    }
    start = 0
  else:
    print(f"Found checkpoint at step {latest} — resuming.")
    state = mngr.restore(latest)
    start = latest + 1

  end = start + 5
  print(f"Training steps {start}..{end - 1}\n")

  for step in range(start, end):
    state["step"] = step
    state["weights"] = jnp.ones((10, 10)) * (step + 1)
    mngr.save(step, state)
    mngr.wait_until_finished()
    print(f"  step {step}: weights mean = {float(state['weights'].mean()):.2f}")
    time.sleep(1)

  return {"completed_step": end - 1, "checkpoint_dir": output_dir}


if __name__ == "__main__":
  print(train_with_auto_resume())
