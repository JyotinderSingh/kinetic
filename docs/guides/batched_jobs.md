# Batched Jobs

This page explains `run_async_map()`, the call that runs one decorated
function over many inputs as independent jobs. Use it for a
hyperparameter sweep, for one job per dataset shard, or for an
evaluation grid. The call returns one `BatchHandle` for the whole
batch. With that handle you watch progress, collect the results in input
order, handle failures, cancel jobs, and delete the resources.

## Before you start

- Read [Detached Jobs](async_jobs.md). Each job in a batch is a normal
  detached job with its own `JobHandle`, its own pod, and its own
  artifacts in the jobs bucket.
- Know the job statuses: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`,
  and `NOT_FOUND`. A batch reports the same statuses per job.

## A first batch

Call `run_async_map()` on a `@kinetic.run()`-decorated function with a
list of inputs. Kinetic submits one job per input and returns a
`BatchHandle`.

```python
import kinetic


@kinetic.run(accelerator="tpu-v5litepod-4")
def train(lr):
  import keras
  import numpy as np

  x = np.random.rand(1000, 20).astype("float32")
  y = x.sum(axis=1, keepdims=True)
  model = keras.Sequential(
    [keras.layers.Dense(64, activation="relu"), keras.layers.Dense(1)]
  )
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
  history = model.fit(x, y, epochs=10, verbose=0)
  return history.history["loss"][-1]


batch = train.run_async_map([0.001, 0.01, 0.1])
losses = batch.results()
print(losses)  # [0.32, 0.28, 0.41] — one result per input, in input order
```

:::{note}
A blocking call to the decorated function blocks until that one job ends.
To run many inputs at the same time, use `run_async_map()`.
:::

`run_async_map()` accepts these keyword arguments:

| Argument | Default | Meaning |
| -------- | ------- | ------- |
| `input_mode` | `"auto"` | How Kinetic passes each item to the function. See [Input modes](#input-modes). |
| `max_concurrent` | `64` | The maximum number of jobs that are active at one time. `None` removes the limit. |
| `retries` | `0` | The number of additional attempts for an input after a job failure. |
| `fail_fast` | `False` | Stop the submission of new jobs after the first failure. |
| `cancel_running_on_fail` | `False` | With `fail_fast=True`, also cancel the running jobs after the first failure. |
| `name`, `tags` | `None` | A name and key-value metadata that Kinetic stores in the batch manifest. |
| `project`, `cluster` | `None` | One-off overrides. Leave them unset; the active profile supplies them. |

## Input modes

The `input_mode` argument controls how Kinetic passes each item in
`inputs` to the function.

| `input_mode`       | Item type                         | Call            | Example item                  |
| ------------------ | --------------------------------- | --------------- | ----------------------------- |
| `"auto"` (default) | `dict` with valid identifier keys | `fn(**item)`    | `{"lr": 0.01, "wd": 1e-4}`    |
| `"auto"` (default) | `list` or `tuple`                 | `fn(*item)`     | `[0.01, 32]`                  |
| `"auto"` (default) | any other type                    | `fn(item)`      | `0.01`                        |
| `"single"`         | any                               | `fn(item)`      | Always one positional argument |
| `"args"`           | `list` or `tuple` (required)      | `fn(*item)`     | `[0.01, 32]`                  |
| `"kwargs"`         | `dict` (required)                 | `fn(**item)`    | `{"lr": 0.01}`                |

### Dict inputs

In `"auto"` mode, Kinetic unpacks a dict with valid Python identifier
keys as keyword arguments:

```python
@kinetic.run(accelerator="tpu-v5litepod-4")
def train(lr, batch_size): ...


configs = [
  {"lr": 0.001, "batch_size": 32},
  {"lr": 0.01, "batch_size": 64},
]
batch = train.run_async_map(configs)
```

### Prevent unpacking

If your function takes a list or a dict as one argument, pass
`input_mode="single"`:

```python
@kinetic.run(accelerator="cpu")
def process(items):
  return sum(items)


batch = process.run_async_map([[1, 2, 3], [4, 5, 6]], input_mode="single")
```

:::{note}
In `"auto"` mode, Kinetic does not unpack every dict. A dict with a key
that is not a valid identifier, for example `{"not-an-id": 1}`, becomes
one positional argument. The same applies to a dict with a key that is
a Python keyword, for example `{"class": 1}`. Use `input_mode="kwargs"`
or `input_mode="single"` if you need explicit control.
:::

## Monitor a batch

You can inspect the batch at any time through the `BatchHandle`.

```python
# Per-job status
for idx, status in batch.statuses():
  print(f"Job {idx}: {status.value}")

# Aggregate counts
print(batch.status_counts())
# {'RUNNING': 2, 'SUCCEEDED': 1}

# Block until every job is terminal (optional timeout in seconds)
batch.wait(timeout=1800)
```

`statuses()` returns `(index, JobStatus)` pairs for each submitted job.
Kinetic skips a slot that is not submitted yet, for example under a
concurrency limit. `wait()` blocks until the submission ends and every
submitted job is terminal. `wait()` raises `TimeoutError` if the
timeout expires.

## Collect results

### `results()`

`results()` is the simplest way to collect every result. It blocks
until every job ends and returns the results in input order.

```python
losses = batch.results()
# losses[0] belongs to inputs[0], losses[1] to inputs[1], and so on
```

**Parameters:**

- **`timeout`** (`float | None`, default `None`): The maximum number of
  seconds to wait. `results()` raises `TimeoutError` when the timeout
  expires.
- **`ordered`** (`bool`, default `True`): `True` returns the results
  aligned with `inputs`. `False` returns the results in the order in
  which the jobs ended.
- **`cleanup`** (`bool`, default `True`): Delete the resources of each
  child after Kinetic downloads its result. See the caution below.
- **`return_exceptions`** (`bool`, default `False`): When `True`, a
  failed position holds the exception object and `results()` does not
  raise. When `False`, any failure raises `BatchError`.

:::{caution}
With the default `cleanup=True`, `results()` deletes the Kubernetes Job
of every child, and also the Cloud Storage artifacts of every child
that succeeded. Those artifacts include the child's `handle.json`, so
`attach_batch()` cannot load those children later and can block. If you
want to reattach to the batch later, or to read the logs of a failed
child, call `results(cleanup=False)`. Call `batch.cleanup()` when you
no longer need the batch. See [Clean up](#clean-up).
:::

:::{important}
A `TimeoutError` does not cancel the jobs. The jobs continue to run on
the cluster. Call `batch.cancel()` if you want to stop them after a
timeout, and read [Manual cancellation](#manual-cancellation) first.
:::

`ordered=False` does not give you earlier access to a result.
`results(ordered=False)` also returns only after every job is terminal.
It changes only the order of the list and the moment at which Kinetic
cleans up each child. To process results as jobs end, use
`as_completed()`.

### `as_completed()`

`as_completed()` yields each `JobHandle` as its job reaches a terminal
state, in completion order.

```python
for job in batch.as_completed():
  result = job.result()
  print(f"{job.job_id} finished: {result}")
```

`as_completed()` yields jobs while the submission of other inputs is
still in progress. Under a concurrency limit, you can process the first
results before Kinetic submits the last inputs. Each `job.result()` call
in the loop cleans up that child by default; pass `cleanup=False` to
keep its resources.

**Parameters:**

- **`poll_interval`** (`float`, default `5.0`): The number of seconds
  between status polls.
- **`timeout`** (`float | None`, default `None`): The maximum number of
  seconds to wait. `as_completed()` raises `TimeoutError` when the
  timeout expires.

## Handle failures

When any job fails and `return_exceptions=False` (the default),
`results()` raises a `BatchError`.

```python
try:
  results = batch.results(cleanup=False)
except kinetic.BatchError as e:
  print(e)  # "Batch grp-a1b2c3d4: 2 of 10 jobs failed"
  for job in e.failures:
    if job is None:
      continue  # This input failed at submission. See batch.submission_failures.
    print(f"{job.job_id}: {job.status().value}")
    print(job.tail(n=20))
  for idx, exc in batch.submission_failures.items():
    print(f"Input {idx} failed at submission: {exc}")
  # e.partial_results holds the result at each successful position
  # and None at each failed position.
```

`BatchError` has three attributes:

- **`group_id`**: The batch identifier.
- **`failures`**: A list with one entry per failed input. The entry is
  the `JobHandle` of the failed job, or `None` if the input failed at
  submission time. Test for `None` before you use the entry.
- **`partial_results`**: With `ordered=True`, a list aligned with
  `inputs`, where a successful position holds the result and a failed
  position holds `None`. With `ordered=False`, a shorter list in
  completion order that holds only the successful results.

The example passes `cleanup=False`. With the default `cleanup=True`,
`results()` deletes the Kubernetes Job of every child before it raises
`BatchError`. After that deletion, `job.status()` returns `NOT_FOUND`
and `job.tail()` raises `RuntimeError`, because the pod is gone. Even
with `cleanup=False`, Kubernetes deletes a finished Job about 10 minutes
after it ends, so read the logs soon after the failure. This retention
window applies to single-host jobs on the GKE backend. A multi-host
Pathways job has no retention window; its resources stay until a
cleanup call deletes them.

### Tolerate failures

Pass `return_exceptions=True` to collect the results without an
exception. A failed position holds the exception object.

```python
results = batch.results(return_exceptions=True)
for i, r in enumerate(results):
  if isinstance(r, Exception):
    print(f"Job {i} failed: {r}")
  else:
    print(f"Job {i}: {r}")
```

### Inspect failed jobs

`failures()` returns the handles of the jobs with status `FAILED`. It
excludes `NOT_FOUND`, because that status is ambiguous. A job can be
`NOT_FOUND` because Kinetic deleted its Kubernetes resources, not
because it failed. Use `statuses()` for a finer inspection. After
`results()` has run, `failures()` returns the list of failed handles
that `results()` recorded.

```python
batch.wait()
for job in batch.failures():
  print(f"{job.job_id}: {job.tail(n=20)}")
```

`job.tail()` reads the pod log, so call it while the pod exists. Call
it after `wait()` and before `results()`, or after
`results(cleanup=False)`, within the 10-minute Kubernetes retention
window.

### Submission failures

The call that submits an input can raise, for example because of a
packaging or validation error. Kinetic then records the exception and
leaves `batch.jobs[idx]` as `None`. Kinetic does not retry a submission
failure. The `batch.submission_failures` property returns a dict that
maps the input index to the exception. `results()` reports these inputs
as failures. The entry in `BatchError.failures` is `None`. The position
in the results holds the exception object only when
`return_exceptions=True`. `wait()` logs a warning when a batch has
submission failures.

## Retries

The `retries` argument sets the number of additional attempts that an
input gets after a job failure. The total number of attempts per input
is `1 + retries`.

```python
batch = train.run_async_map(configs, retries=2)
# Each input gets up to 3 attempts (1 initial + 2 retries)
```

- Kinetic starts a retry when a job reaches `FAILED` or `NOT_FOUND`.
- Before each retry, Kinetic deletes the Kubernetes resources of the
  previous attempt and keeps its Cloud Storage artifacts.
- Each attempt is a new job with a new job ID. The batch manifest keeps
  only the latest job ID for the input.
- Kinetic does not retry a submission failure.

:::{note}
When `retries > 0`, Kinetic runs the submission loop in a background
thread, so that it can poll for failures and resubmit.
:::

## Concurrency control

By default, `run_async_map()` limits the number of active jobs to 64.
Use `max_concurrent` to change the limit.

```python
# At most 8 jobs run at one time
batch = train.run_async_map(configs, max_concurrent=8)
```

```python
# Submit every job at once (no limit)
batch = train.run_async_map(configs, max_concurrent=None)
```

- **Default `64`:** Kinetic starts a new job each time an active job
  ends.
- **`None`:** Kinetic submits every input at once. With `retries=0`
  (the default), the submission runs in the calling thread before
  `run_async_map()` returns.
- The value must be a positive integer when set. `0` or a negative
  value raises `ValueError`.

:::{note}
Kinetic logs a warning when you submit more than 100 inputs with
`max_concurrent=None`. Set a limit to control the resource usage.
:::

## Cancellation and fail-fast

### Fail-fast behavior

The `fail_fast` and `cancel_running_on_fail` arguments control what
happens when a job fails.

| `fail_fast`       | `cancel_running_on_fail` | On the first failure                                                              |
| ----------------- | ------------------------ | --------------------------------------------------------------------------------- |
| `False` (default) | `False` (default)        | All remaining jobs continue. Kinetic reports the failures at the end.             |
| `True`            | `False`                  | Kinetic starts no new jobs. Jobs that already run continue to the end.            |
| `True`            | `True`                   | Kinetic starts no new jobs and cancels all running jobs at once.                  |
| `False`           | `True`                   | **No effect.** `cancel_running_on_fail` applies only when `fail_fast=True`.       |

```python
# Stop the batch as soon as any job fails, and cancel all running jobs
batch = train.run_async_map(
  configs,
  fail_fast=True,
  cancel_running_on_fail=True,
)
```

A "failure" here is either a submission failure or a runtime failure: a
job that reaches `FAILED` or `NOT_FOUND` after all its attempts.

:::{note}
`run_async_map(max_concurrent=None, retries=0, fail_fast=True)` does not
return at once. In that configuration the submission loop runs in the
calling thread, and `fail_fast` makes the loop poll until every job is
terminal. If you want the call to return at once, set a concurrency
limit or leave `fail_fast=False`.
:::

### Manual cancellation

`batch.cancel()` cancels every submitted job that is not terminal:

```python
batch.cancel()
```

Cancellation deletes the Kubernetes resource of each job and keeps the
Cloud Storage artifacts. Cancellation does not stop the background
submission loop. Two consequences follow:

- Under a concurrency limit, the loop sees the cancelled jobs as
  terminal, frees their slots, and submits the remaining inputs.
- With `retries > 0`, the loop treats each cancelled job as a failed
  attempt and resubmits the input until no attempts remain.

To make `batch.cancel()` stop the whole batch, use one of these
configurations:

- `max_concurrent=None` with the defaults `retries=0` and
  `fail_fast=False`. Kinetic submits every input before
  `run_async_map()` returns, so no input remains for the loop to
  submit.
- `fail_fast=True` with `retries=0`. The loop treats the first cancelled
  job as a failure and stops the submission of new inputs.

## Reattach to a batch

If your local process exits, or if you want to check a batch from a
different machine, save the `group_id` and reattach later.

```python
# Original session
batch = train.run_async_map(configs)
print(f"Batch ID: {batch.group_id}")  # e.g., "grp-a1b2c3d4"
results = batch.results(cleanup=False)  # keep the child handles

# Later, from any machine with the same active profile
batch = kinetic.attach_batch("grp-a1b2c3d4", poll_timeout=60)
results = batch.results(cleanup=False)
batch.cleanup()  # when you are done
```

`attach_batch()` downloads the batch manifest from Cloud Storage and
rebuilds a `JobHandle` for each child. Kinetic keeps the index
alignment. If the original batch had 10 inputs and the process crashed
after 7 submissions, `batch.jobs` has 10 entries. The 3 slots without a
submission hold `None`.

When the manifest has fewer children than expected, `attach_batch()`
logs a warning and starts a background thread. That thread polls the
manifest until all children appear or until `poll_timeout` expires.
`wait()`, `results()`, and `as_completed()` block until that thread
ends. The default `poll_timeout=None` polls without a limit.

:::{caution}
In two situations, `attach_batch()` loads fewer children than expected,
and the missing children never appear. First, the original process
crashed during the submission. Second, `results()` with the default
`cleanup=True` deleted the `handle.json` of the successful children. In
both situations, `attach_batch()` with the default `poll_timeout=None`
makes a later `wait()`, `results()`, or `as_completed()` block and never
return. Pass `poll_timeout` when you reattach. Use
`results(cleanup=False)` in the original session if you plan to
reattach.
:::

**Parameters:**

- **`group_id`** (`str`): The batch identifier (for example
  `"grp-a1b2c3d4"`).
- **`project`** (`str | None`, default `None`): A one-off override. The
  active profile supplies the project when `None`.
- **`cluster`** (`str | None`, default `None`): A one-off override. The
  active profile supplies the cluster when `None`.
- **`poll_interval`** (`float`, default `10.0`): The number of seconds
  between manifest polls when children are missing.
- **`poll_timeout`** (`float | None`, default `None`): The maximum
  number of seconds to poll for missing children. `None` polls without
  a limit.

## Clean up

There are two ways to delete the resources of a batch.

### Automatic cleanup through `results()`

By default, `results()` cleans up each child after it downloads the
result of that child. For every child, Kinetic deletes the Kubernetes
Job. For a child that succeeded, Kinetic also deletes the Cloud Storage
artifacts, including `handle.json`. A child that failed keeps its
artifacts. Kinetic keeps the batch manifest.

```python
results = batch.results()  # cleanup=True is the default
```

Because the successful children lose their `handle.json`, a later
`attach_batch()` finds the manifest but cannot load those children. Use
`results(cleanup=False)` if you plan to reattach, then call
`batch.cleanup()` at the end.

### Full cleanup

To delete everything, including the batch manifest, call `cleanup()` on
the handle:

```python
batch.cleanup(k8s=True, gcs=True)
```

**Parameters:**

- **`k8s`** (`bool`, default `True`): Delete the Kubernetes resources
  (Jobs and pods) of each child.
- **`gcs`** (`bool`, default `True`): Delete the Cloud Storage
  artifacts of each child **and** the batch manifest.

:::{important}
After `cleanup(gcs=True)`, `attach_batch()` cannot find the batch,
because the manifest no longer exists.
:::

## How it works

### Threading model

When `max_concurrent` is set (the default is 64) or `retries > 0`,
`run_async_map()` starts a background thread that manages the
submission. The thread polls the active jobs for terminal states and
starts new jobs when slots become free. `run_async_map()` returns the
`BatchHandle` at once. The thread is not a daemon thread, so the Python
process stays alive until the submission ends.

When `max_concurrent=None` and `retries=0`, Kinetic submits every input
in the calling thread and starts no background thread. With
`fail_fast=False`, `run_async_map()` returns as soon as the submission
ends. With `fail_fast=True`, the calling thread also polls until every
job is terminal, so `run_async_map()` returns only when the batch ends.

### Manifest

Kinetic writes a JSON manifest to `gs://{jobs bucket}/_groups/{group_id}/manifest.json`
before it submits the first job. The manifest records the batch metadata
(group ID, expected total, function name, name, and tags). Kinetic
updates the manifest after each successful submission with the child's
index and job ID. `attach_batch()` reads the manifest to find the
submitted jobs and rebuilds the handle from each child's `handle.json`.

### Group ID

Each batch gets a unique identifier in the format `grp-{8 hex chars}`
(for example `grp-a1b2c3d4`). Kinetic sets this ID on each child
`JobHandle` as `group_id`, together with `group_kind="map"` and the
child's `group_index`.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em` Detached Jobs
:link: async_jobs
:link-type: doc

The `run_async()` workflow and the `JobHandle` that each child of a batch uses.
:::

:::{grid-item-card} {octicon}`zap;1em` Cost Optimization
:link: cost_optimization
:link-type: doc

A batch multiplies the cost as well as the throughput; concurrency limits and Spot capacity help.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: checkpointing
:link-type: doc

Each child writes to its own `KINETIC_OUTPUT_DIR`; useful for long work inside a batch.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

What to do when children stay in `PENDING` or fail repeatedly.
:::
::::
