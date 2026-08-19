# Detached Jobs

A blocking call to a decorated function blocks your local process until the
job ends. That is the right choice for a short job or for interactive
work. A **detached job** is a job that you submit with `run_async()`. The
call returns a `JobHandle` at once, and the job runs on the cluster
without your local process. You then check the status, read the logs,
collect the result, and delete the job. You do this from Python or from
the `kinetic jobs` command group, on any machine.

This page covers the loop from submit to cleanup. It shows a first
example, the Python and CLI operations side by side, the job lifecycle,
and how to reattach from another machine. It ends with timeouts, cleanup,
and recommendations for long jobs.

## Before you start

- Complete [Getting Started](../getting_started.md). The active profile
  supplies the project, the zone, the cluster, and the namespace for every
  call and command on this page.
- Read [How Kinetic Works](../concepts.md) for the vocabulary: job, job
  ID, pod, jobs bucket, and output directory.

## A first detached job

```python
import kinetic


@kinetic.run(accelerator="tpu-v5litepod-4")
def train_model():
  # Long-running training code
  return {"final_loss": 0.123}


job = train_model.run_async()
print(f"Submitted: {job.job_id}")  # for example: job-3f9a1c2b

# ... do other work, or exit this script ...

final = job.result(timeout=3600)  # blocks until the job ends, or for 3600 s
print(final)
```

`@kinetic.run()` takes the same arguments for a blocking call and for a
detached job: `accelerator`, `volumes`, `capture_env_vars`, `output_dir`,
and the others. Only the call changes. `train_model()` blocks and returns
the value. `train_model.run_async()` returns a `JobHandle`.

`run_async()` returns after Kinetic packages your code, builds or reuses
the container image, uploads the artifacts, and creates the Kubernetes
Job. If Kinetic must build a new image, `run_async()` blocks for the
build, which takes about 5 to 10 minutes. After `run_async()` returns,
the job does not need your local process. You can close the script.

The job ID has the form `job-` plus 8 hexadecimal characters, for example
`job-3f9a1c2b`. Read it from `job.job_id`, as the example does. Save the
ID. With the active profile, the ID is all that you need to reattach.

## Python and CLI side by side

Each operation after submit exists as a `JobHandle` method and as a
`kinetic jobs` subcommand. Use the one that fits your workflow.

Operation        | Python                            | CLI
---------------- | --------------------------------- | ----------------------------------------------
Submit           | `job = train_model.run_async()`   | (no CLI command; call `run_async()` from a script)
Reattach         | `job = kinetic.attach(job_id)`    | (pass `<id>` to any `kinetic jobs` subcommand)
List             | `kinetic.list_jobs()`             | `kinetic jobs list`
Check status     | `job.status()`                    | `kinetic jobs status <id>`
Read all logs    | `job.logs()`                      | `kinetic jobs logs <id>`
Tail logs        | `job.tail(n=100)`                 | `kinetic jobs logs <id> --tail 100` (or `-n 100`)
Follow logs      | `job.logs(follow=True)`           | `kinetic jobs logs <id> --follow` (or `-f`)
Wait for result  | `job.result(timeout=3600)`        | `kinetic jobs result <id> --timeout 3600`
Cancel           | `job.cancel()`                    | `kinetic jobs cancel <id>`
Clean up         | `job.cleanup(k8s=True, gcs=True)` | `kinetic jobs cleanup <id>`

`--follow` and `--tail` are exclusive. `kinetic jobs logs` rejects a
command that has both flags. Without a flag, the command prints the full
log of the pod.

Both the Python functions and the CLI read the active profile. See
[Where `attach()` and the CLI find the cluster](#where-attach-and-the-cli-find-the-cluster).

## Job lifecycle

A job moves through five states. `JobStatus` in `kinetic.job_status`
defines them.

```text
                  ┌──────────┐
 run_async() ───▶ │ PENDING  │ ── the Job exists, no pod runs yet
                  └────┬─────┘
                       │ pod scheduled and started
                       ▼
                  ┌──────────┐
                  │ RUNNING  │ ── the pod runs
                  └────┬─────┘
              ┌────────┴────────┐
              ▼                 ▼
        ┌───────────┐     ┌──────────┐
        │ SUCCEEDED │     │  FAILED  │
        └───────────┘     └──────────┘

  NOT_FOUND ── the Kubernetes resource no longer exists (deleted by
               result(), cancel(), cleanup(), or the 10-minute timer)
```

What each state means, and what to do:

- **PENDING** — Kubernetes accepted the Job, but no pod runs yet. The
  cluster autoscaler starts a node if none is free. On a node pool that
  is scaled to zero, the node start takes about 2 to 5 minutes. *What to
  do:* wait. If the job stays `PENDING` for more than 10 minutes, run
  `kinetic pool list` and make sure that a node pool for the accelerator
  exists. Then run `kinetic init`, select `troubleshoot`, and check the
  accelerator quota of the project. See
  [Troubleshooting](../troubleshooting.md#a-job-stays-in-pending-for-more-than-10-minutes).
- **RUNNING** — the pod runs. The pod first downloads the artifacts and
  then calls your function. *What to do:* nothing. Use `job.tail()` or
  `kinetic jobs logs <id> -f` to watch the progress.
- **SUCCEEDED** — your function returned, the pod uploaded the return
  value, and the pod exited. *What to do:* call `job.result()` to get the
  return value. With the default cleanup, `result()` also deletes the
  Kubernetes Job and the Cloud Storage artifacts.
- **FAILED** — your function raised an exception, or the pod exited with
  a non-zero code. *What to do:* read the logs first, with `job.tail()`
  or `kinetic jobs logs <id>`. Then call `job.result()`. `result()`
  raises the remote exception with the remote traceback. With the
  default cleanup, `result()` deletes the Kubernetes Job and its pod for
  a failed job too, so the logs are gone after that call. The GCS
  artifacts of a failed job stay. See
  [Cleanup and what remains](#cleanup-and-what-remains).
- **NOT_FOUND** — the Kubernetes Job no longer exists. Four things cause
  this state:
  - a `result()` call with the default cleanup, on success and on
    failure;
  - a `cancel()` call;
  - a `cleanup()` call;
  - the Kubernetes timer that deletes a finished Job 10 minutes after
    the job ends (2 hours for a job with `debug=True`); a multi-host
    TPU job has no timer.

  A job that you check one hour after it ended is therefore `NOT_FOUND`.
  That state is normal. *What to do:* if you need the return value, call
  `result()` one time. `result()` reads the result from Cloud Storage when the
  artifacts still exist, and returns the value or raises the remote
  exception. If `result()` raises `RuntimeError` with "no result payload
  exists", the artifacts are gone and the job is not recoverable.

The full flow from submit to cleanup:

:::{container} kinetic-steps
1. **Submit.**
   `run_async()` packages your code and builds or reuses the container
   image. It uploads the artifacts and a `handle.json` file to
   `gs://{jobs bucket}/{job_id}/`. Then it creates a Kubernetes Job and
   returns a `JobHandle`. The status is `PENDING`.
2. **Schedule.**
   The cluster autoscaler starts a node if none is free. Kubernetes
   schedules the pod. The status changes to `RUNNING`.
3. **Run.**
   The pod downloads the artifacts and calls your function. When the
   function returns or raises, the pod uploads the return value or the
   exception to `gs://{jobs bucket}/{job_id}/result.pkl` and exits.
4. **Finish.**
   The status changes to `SUCCEEDED` or `FAILED`. Kubernetes starts the
   10-minute timer.
5. **Collect and clean up.**
   `job.result()` downloads `result.pkl`, returns the value or raises the
   exception, and deletes the Kubernetes Job. On success, `result()` also
   deletes the Cloud Storage artifacts. The status is now `NOT_FOUND`. A second
   `result()` call on a successful job fails, because the artifacts are
   gone.
:::

## Reattach from another machine

At submit time, Kinetic writes the `JobHandle` as a small JSON file to
the jobs bucket. `kinetic.attach(job_id)` reads that file and rebuilds
the handle. Reattach works from any machine that has Kinetic installed,
Google Cloud credentials for the same project, and a profile for the same
cluster:

```python
import kinetic

job = kinetic.attach("job-3f9a1c2b")
print(f"Status: {job.status().value}")
print(job.tail(n=20))
```

If you do not remember the ID, list the live jobs on the cluster:

```python
for j in kinetic.list_jobs():
  print(f"{j.job_id}  {j.func_name}  {j.accelerator}  {j.status().value}")
```

The CLI equivalent is `kinetic jobs list`. It prints the job ID, the
function name, the accelerator, the backend, and the creation time.

Two limits apply:

- `list_jobs()` and `kinetic jobs list` show only jobs whose Kubernetes
  resource still exists. A job that is `NOT_FOUND` does not appear in
  the list. Keep the job ID if you need the job later.
- A `result()` call with the default cleanup deletes the whole
  `gs://{jobs bucket}/{job_id}/` prefix on success, including
  `handle.json`. After that, `attach(job_id)` fails because the file no
  longer exists.

## Where `attach()` and the CLI find the cluster

The jobs bucket is `gs://{project}-kn-{cluster}-jobs`, so `attach()` and
`list_jobs()` need the project and the cluster name to find a job. Both
functions accept `project=` and `cluster=` (`list_jobs()` also accepts
`zone=` and `namespace=`). Kinetic resolves each value in this order, and
the first value wins:

1. The keyword argument, for example `kinetic.attach(job_id, cluster="research")`.
2. The `KINETIC_*` environment variable, for example `KINETIC_CLUSTER`.
3. The active profile.
4. The built-in default.

The `kinetic jobs` command group uses the same order. The flag
(`--project`, `--zone`, `--cluster`, and `--namespace` for `list`) wins
over the environment variable. The environment variable wins over the
active profile. To run one command against a different profile, put
`--profile NAME` before the subcommand:

```bash
kinetic --profile research jobs list
```

With the active profile from `kinetic init`, you pass no flags. See
[Profiles](profiles.md).

## Timeouts

`result()` blocks until the job ends. Pass `timeout=` (in seconds) to
bound the wait:

```python
try:
  final = job.result(timeout=3600)
except TimeoutError:
  # The job still runs, and the handle is still valid. You can call
  # result() again, tail(), or cancel(). Or you can exit the script.
  print(job.tail(n=50))
```

A `TimeoutError` does not stop the job. It only returns control to your
script. `result()` polls the status every 5 seconds. Pass
`stream_logs=True` to print the pod log to your terminal while `result()`
waits.

## Cleanup and what remains

Three things belong to a job: the Kubernetes Job with its pod, the GCS
artifacts under `gs://{jobs bucket}/{job_id}/`, and the files that your
function wrote under `KINETIC_OUTPUT_DIR`. Cleanup touches the first two
only.

`result()` deletes the Kubernetes Job by default. It does this in every
case: on success and on failure. `result()` deletes the Cloud Storage artifacts
only when it collected a result, that is, on success. The artifacts of a
failed job stay in the bucket, so `result()` can raise the same remote
exception again from another machine.

To keep the Kubernetes Job and the pod log after `result()`, pass
`cleanup=False`. Delete the Job later with `cleanup()`:

```python
final = job.result(cleanup=False)  # keep the Job and the artifacts
job.cleanup(k8s=True, gcs=False)  # later: delete the Job, keep the artifacts
```

To read the logs of a failed job, do one of these:

- Read the logs before you call `result()`, with `job.tail()`,
  `job.logs()`, or `kinetic jobs logs <id>`.
- Call `job.result(cleanup=False)`, read the logs, and call
  `job.cleanup()` when you are done.

:::{note}
If nobody calls `result()`, `cancel()`, or `cleanup()`, Kubernetes
deletes a finished Job and its pod without a call from you: 10 minutes after the job
ends, or 2 hours after the end for a job with `debug=True`. The pod log
is deleted with the pod. The Cloud Storage artifacts stay until you delete them or until
the 30-day rule of the jobs bucket deletes them. A multi-host TPU job
uses a LeaderWorkerSet resource, and Kinetic sets no timer on that
resource. Call `result()`, `cancel()`, or `cleanup()` to delete it. See
[Distributed Training](distributed_training.md).
:::

The cleanup operations in detail:

- `job.cancel()` deletes the Kubernetes Job and its pod, and keeps the
  Cloud Storage artifacts. Use it to stop a running job. The status becomes
  `NOT_FOUND`.
- `job.cleanup(k8s=True, gcs=True)` deletes one or both parts. Set
  `k8s=False` or `gcs=False` to keep a part.
- A job with `debug=True` keeps its Kubernetes Job after `result()`
  because `cleanup` defaults to `False` for debug jobs. See
  [Interactive Debugging](debugging.md).

The CLI has the same options:

```bash
kinetic jobs result <id> --no-cleanup   # collect, keep the Job and the artifacts
kinetic jobs cleanup <id> --no-gcs      # delete the Job, keep the artifacts
kinetic jobs cleanup <id> --no-k8s      # delete the artifacts, keep the Job
kinetic jobs cancel <id>                # stop the job, keep the artifacts
```

`kinetic jobs result`, `cancel`, and `cleanup` also accept
`--cleanup-timeout` (default 180 seconds) and `--cleanup-poll-interval`
(default 2 seconds). Kinetic waits up to `--cleanup-timeout` for
Kubernetes to confirm the deletion.

Kinetic never deletes the files under `KINETIC_OUTPUT_DIR` as part of
job cleanup. The default output directory is
`gs://{jobs bucket}/outputs/{job_id}`, which is outside the
`{job_id}/` prefix. The 30-day rule of the jobs bucket applies to those
files. See [Outputs and Checkpoints](checkpointing.md).

## Recommendations for long jobs

These practices reduce the cost of a failure in a job that runs for hours.

- **Write checkpoints at a regular interval.** The files under
  `KINETIC_OUTPUT_DIR` survive a failed pod, but a restart can use only
  the checkpoints that exist. Select an interval that bounds the work
  that a restart loses. See [Outputs and Checkpoints](checkpointing.md)
  for the resume pattern.
- **Save the job ID.** Print it, write it to a log file, or record it in
  your experiment tracker. With the ID, you reattach from any machine
  that has Kinetic, credentials for the project, and a profile for the
  cluster.
- **Do not depend on the local Python process.** After `run_async()`
  returns, the local script has no part in the job. If you stop the
  script, for example with `Ctrl-C`, the remote job continues.
- **Do not follow the logs of a job that runs for hours.** A log stream
  breaks on a short network failure. Read the last lines from a new shell
  with `kinetic jobs logs <id> -n 200` instead, at the interval that you
  choose.
- **Keep the artifacts of a multi-host or high-cost job.** Pass
  `cleanup=False` to the first `result()` call, so that the Kubernetes
  resources and the Cloud Storage artifacts stay for inspection. Call `cleanup()`
  when you no longer need them.
- **Read the logs of a failed job before you collect it.** A `result()`
  call with the default cleanup deletes the pod, and the pod log with it.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: checkpointing
:link-type: doc

Write durable outputs and make a long job resumable.
:::

:::{grid-item-card} {octicon}`stack;1em` Batched Jobs
:link: batched_jobs
:link-type: doc

Run one function over many inputs with `run_async_map()`.
:::

:::{grid-item-card} {octicon}`zap;1em` Cost Optimization
:link: cost_optimization
:link-type: doc

Spot capacity and scale-to-zero behavior for detached jobs.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

What to do when a job stays in `PENDING` or fails repeatedly.
:::
::::
