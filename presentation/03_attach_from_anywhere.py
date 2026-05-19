"""Demo 3b — Reattach from a fresh shell with just the job id.

This script imports kinetic and calls `kinetic.attach(job_id)`.
That's all it takes — no shared state with the submitting process,
no pickle file, no env var. The cluster is the source of truth.

Run in terminal 2 after 05_submit_detached.py:
    python 06_attach_from_anywhere.py <job_id>

Talk track:
    "Different process. Different shell. Could be a different machine.
     I have a job id and that's it. kinetic.attach() reconstructs
     everything I need: status, logs, the result when it lands."
"""

import sys

import kinetic


def main():
  if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <job_id>")
    sys.exit(1)

  job_id = sys.argv[1]
  print(f"Attaching to {job_id} ...\n")

  handle = kinetic.attach(job_id)

  print(f"  function:    {handle.func_name}")
  print(f"  backend:     {handle.backend}")
  print(f"  accelerator: {handle.accelerator}")
  print(f"  status:      {handle.status().value}\n")

  print("--- Last 20 log lines ---")
  print(handle.tail(n=20))
  print("-------------------------\n")

  print("Blocking on result (streaming live logs)...")
  result = handle.result(stream_logs=True, cleanup=False)
  print(f"\nResult: {result}")

  handle.cleanup()
  print(f"\nCleaned up {job_id}.")


if __name__ == "__main__":
  main()
