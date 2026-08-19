# Batched Jobs

`run_async()` is the tool for a single long-running job. When you
need to run the **same function over many inputs**, such as a hyperparameter
sweep, one job per dataset shard, an evaluation grid — wiring that up
by hand means a loop that calls `run_async()`, your own bookkeeping for
which handles are still live, your own error aggregation, your own
cleanup. `run_async_map()` is that loop, done for you.

You call `run_async_map()` on a `@kinetic.run()`-decorated function with a list of
inputs. It returns a single `BatchHandle` that represents the whole
collection: one place to observe progress, collect results in input
order, handle failures, cancel siblings, and tear everything down. The
underlying jobs are independent Kinetic jobs — each one gets a real
`JobHandle`, runs on its own pod, and writes its own artifacts to GCS.

This page builds on the single-job workflow covered in
[Detached Jobs](async_jobs.md). Familiarity with `JobHandle` and the
`PENDING`/`RUNNING`/`SUCCEEDED`/`FAILED`/`NOT_FOUND` lifecycle is
assumed.

## A first fan-out

Pass a `@kinetic.run()`-decorated function and a list of inputs to
`run_async_map()`. It returns a `BatchHandle` immediately while jobs are
submitted in the background.

```python
import kinetic


@kinetic.run(accelerator="tpu-v5e-1")
def train(lr):
  import keras

  model = keras.Sequential(
    [keras.layers.Dense(64, activation="relu"), keras.layers.Dense(1)]
  )
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
  history = model.fit(x_train, y_train, epochs=10, verbose=0)
  return history.history["loss"][-1]


batch = train.run_async_map([0.001, 0.01, 0.1])
losses = batch.results()
print(losses)  # [0.32, 0.28, 0.41] — one result per input, in order
```

:::{note}
You must use `run_async_map()` to fan out. Calling the decorated function
directly will block until the job finishes and return the result directly,
so it cannot be used for concurrent execution of multiple inputs.
:::

## Input modes

The `input_mode` parameter controls how each item in `inputs` is passed
to the function.

| `input_mode`       | Item type                         | How it's called | Example item                  |
| ------------------ | --------------------------------- | --------------- | ----------------------------- |
| `"auto"` (default) | `dict` with valid identifier keys | `fn(**item)`    | `{"lr": 0.01, "wd": 1e-4}`    |
| `"auto"` (default) | `list` or `tuple`                 | `fn(*item)`     | `[0.01, 32]`                  |
| `"auto"` (default) | anything else                     | `fn(item)`      | `0.01`                        |
| `"single"`         | any                               | `fn(item)`      | always passed as a single arg |
| `"args"`           | `list` or `tuple` (required)      | `fn(*item)`     | `[0.01, 32]`                  |
| `"kwargs"`         | `dict` (required)                 | `fn(**item)`    | `{"lr": 0.01}`                |

### Dict inputs (kwargs unpacking)

When using `"auto"` mode, dicts with valid Python identifier keys are
unpacked as keyword arguments:

```python
@kinetic.run(accelerator="tpu-v5e-1")
def train(lr, batch_size): ...


configs = [
  {"lr": 0.001, "batch_size": 32},
  {"lr": 0.01, "batch_size": 64},
]
batch = train.run_async_map(configs)
```

### Preventing unpacking

If your function takes a list or dict as a single argument, use
`input_mode="single"` to prevent automatic unpacking:

```python
@kinetic.run(accelerator="cpu")
def process(items):
  return sum(items)


batch = process.run_async_map([[1, 2, 3], [4, 5, 6]], input_mode="single")
```

:::{note}
In `"auto"` mode, dicts with non-identifier keys (like
`{"not-an-id": 1}`) or Python keywords (like `{"class": 1}`) are passed
as a single positional argument rather than unpacked. Use
`input_mode="kwargs"` or `input_mode="single"` if you need explicit
control.
:::

## Monitoring a batch

You can inspect progress at any time through the `BatchHandle`.

```python
# Per-job status
for idx, status in batch.statuses():
  print(f"Job {idx}: {status.value}")

# Aggregate counts
print(batch.status_counts())
# {'RUNNING': 2, 'SUCCEEDED': 1}
```

`statuses()` returns `(index, JobStatus)` pairs for each submitted job.
Slots that haven't been submitted yet (when using bounded concurrency)
are skipped. Job statuses follow the same lifecycle as single jobs —
see [Detached Jobs](async_jobs.md) for details on `PENDING`, `RUNNING`,
`SUCCEEDED`, `FAILED`, and `NOT_FOUND`.

## Collecting results

### `results()`

The simplest way to collect all results. By default it blocks until
every job finishes and returns results in input order.

```python
# Input order (default)
losses = batch.results()
# losses[0] corresponds to inputs[0], losses[1] to inputs[1], etc.
```

For faster access to early finishers, use `ordered=False` to collect in
completion order:

```python
losses = batch.results(ordered=False)
# Results appear in the order jobs finish, not input order
```

**Parameters:**

- **`timeout`** (`float | None`, default `None`): Maximum seconds to
  wait. Raises `TimeoutError` if exceeded.
- **`ordered`** (`bool`, default `True`): `True` returns results aligned
  with `inputs`. `False` returns results in the order jobs complete.
- **`cleanup`** (`bool`, default `True`): Delete each child's Kubernetes
  resources and GCS artifacts after downloading its result. The group
  manifest is preserved so `attach_batch()` still works.
- **`return_exceptions`** (`bool`, default `False`): When `True`, failed
  positions contain the exception object instead of raising
  `BatchError`. When `False`, any failure raises `BatchError`. A job
  that fails and an input that fails at submission time both count as a
  failure.

:::{important}
A `TimeoutError` does not cancel running jobs. They continue executing
on the cluster. Call `batch.cancel()` explicitly if you want to stop
them after a timeout.
:::

### `as_completed()`

For processing results incrementally as jobs finish, use the
`as_completed()` iterator. It yields `JobHandle` objects in completion
order.

```python
for job in batch.as_completed():
  result = job.result()
  print(f"{job.job_id} finished: {result}")
```

`as_completed()` streams results even while submission is still in
progress. With bounded concurrency, you can start processing the first
results before the last inputs have been submitted.

**Parameters:**

- **`poll_interval`** (`float`, default `5.0`): Seconds between status
  polls.
- **`timeout`** (`float | None`, default `None`): Maximum seconds to
  wait. Raises `TimeoutError` if exceeded.

## Handling failures

When any job fails and `return_exceptions=False` (the default),
`results()` raises a `BatchError`. An input that fails at submission
time raises a `BatchError` too.

```python
try:
  results = batch.results()
except kinetic.BatchError as e:
  print(e)  # Batch grp-a1b2c3d4: 2 of 8 jobs failed
  for job in e.failures:
    print(f"  job {job.job_id}: {job.status().value}")
  for index, exc in e.submission_failures.items():
    print(f"  input {index} never started: {exc}")
  # e.partial_results has results at successful positions, None at failed ones
```

`BatchError` provides four attributes:

- **`group_id`**: The batch identifier.
- **`failures`**: A list of `JobHandle` objects for the jobs that
  started and then failed. The list holds only `JobHandle` objects, so
  `job.job_id` and `job.status()` are always safe to call.
- **`submission_failures`**: A dict that maps an input index to the
  exception from the submission of that input. These inputs never
  became jobs. They have no `JobHandle`, and they never appear in
  `failures`.
- **`partial_results`**: A list aligned with `inputs`. A successful
  position holds the result. A failed position holds `None`.

:::{note}
`partial_results` aligns with `inputs` only for the default
`ordered=True`. With `ordered=False`, it holds the results that
`results()` collected, in completion order.
:::

### Tolerating failures

Use `return_exceptions=True` to collect results without raising. Failed
positions contain the exception object.

```python
results = batch.results(return_exceptions=True)
for i, r in enumerate(results):
  if isinstance(r, Exception):
    print(f"Job {i} failed: {r}")
  else:
    print(f"Job {i}: {r}")
```

### Inspecting failures

`failures()` returns handles for the jobs with status `FAILED`. It
excludes `NOT_FOUND`, because that status is ambiguous. A job can be
`NOT_FOUND` because Kinetic cleaned up its Kubernetes resources, and
not because the job failed. Use `statuses()` for a more exact view.

After `results()` runs, `failures()` returns the failures from that
collection pass, and not the live status of each job. This keeps the
list correct after `cleanup=True` deletes the Kubernetes resources.

```python
for job in batch.failures():
  print(f"{job.job_id}: {job.tail(n=20)}")
```

`failures()` reports only the jobs that started. To see the inputs that
failed before they became jobs, read `submission_failures`. It maps the
input index to the exception from that submission.

```python
for index, exc in batch.submission_failures.items():
  print(f"input {index} failed to submit: {exc}")
```

## Retries

The `retries` parameter specifies how many additional attempts a job
gets after failure. The total number of attempts per input is
`1 + retries`.

```python
batch = train.run_async_map(configs, retries=2)
# Each job gets up to 3 attempts (1 initial + 2 retries)
```

- Retries are triggered when a job reaches `FAILED` or `NOT_FOUND`
  status.
- Before each retry, Kinetic cleans up the previous attempt's
  Kubernetes resources (GCS artifacts are preserved for debugging).
- The group manifest tracks the attempt count per job, so
  `attach_batch()` can distinguish retries from initial submissions.
- Kinetic does not retry a submission error, which is an error that the
  call to the function raises. These errors are usually packaging errors
  or configuration errors, and they fail again.
- Kinetic does not retry a cancelled job. `cancel()` marks its children,
  so the `NOT_FOUND` status that cancellation causes never starts a new
  attempt.

:::{note}
When `retries > 0`, job submission runs in a background thread. This
lets Kinetic poll for failures and submit the input again.
:::

## Concurrency control

By default, `run_async_map()` limits the number of concurrently active
jobs to 64. Use `max_concurrent` to tune this.

```python
# At most 8 jobs running at once
batch = train.run_async_map(configs, max_concurrent=8)
```

```python
# Submit all jobs immediately (no concurrency limit)
batch = train.run_async_map(configs, max_concurrent=None)
```

- **Default:** `64`. Kinetic launches a new job each time a running job
  finishes.
- **`None`:** Kinetic submits all inputs immediately, with no
  concurrency limit. The calling thread does this work when `retries=0`
  and when `fail_fast` and `cancel_running_on_fail` are not both `True`.
  See [Threading model](#threading-model).
- Must be a positive integer when set. Passing `0` or a negative value
  raises `ValueError`.

In every case `run_async_map()` returns the `BatchHandle` as soon as the
submission work is handed off or complete. It never waits for the jobs
to finish. Use `wait()` or `results()` when you want to block.

:::{note}
Kinetic logs a warning when submitting more than 100 jobs with
`max_concurrent=None`, suggesting you set a limit to control resource
usage.
:::

## Cancellation and fail-fast

### Fail-fast behavior

The `fail_fast` and `cancel_running_on_fail` parameters control what
happens when a job fails.

| `fail_fast`       | `cancel_running_on_fail` | On first failure...                                                              |
| ----------------- | ------------------------ | -------------------------------------------------------------------------------- |
| `False` (default) | `False` (default)        | All remaining jobs continue. Failures are collected at the end.                  |
| `True`            | `False`                  | No new jobs are launched. Already-running jobs continue to completion.           |
| `True`            | `True`                   | No new jobs are launched. All running siblings are cancelled immediately.        |
| `False`           | `True`                   | **No effect.** `cancel_running_on_fail` only takes effect when `fail_fast=True`. |

```python
# Stop the batch as soon as any job fails, cancel all running siblings
batch = train.run_async_map(
  configs,
  fail_fast=True,
  cancel_running_on_fail=True,
)
```

A failure is one of two events. The first is a submission error, when
the call raises. The second is a runtime failure, when the remote job
reaches `FAILED` or `NOT_FOUND` status after all of its attempts.

### Manual cancellation

`cancel()` stops the full collection at any time. It is independent of
the `fail_fast` setting.

```python
batch.cancel()
```

`cancel()` does three things:

- It deletes the Kubernetes resource of each job that is not terminal.
  The GCS artifacts of that job stay in place for debugging.
- It drops the inputs that `max_concurrent` holds in the queue. Kinetic
  does not launch them.
- It marks the children as cancelled. Kinetic does not submit them
  again, even when `retries` is above zero.

A cancelled job reports the status `NOT_FOUND`, because its Kubernetes
resource is gone. `wait()` returns after each job that started is
terminal, and the slot of an input that never launched stays `None`.

A cancelled job has no result. `results()` therefore raises a
`BatchError` that lists those jobs in `failures`. Use
`results(return_exceptions=True)` to read the results of the children
that finished before the cancellation.

## Reattaching to a batch

If your local process exits or you want to check on a batch from a
different machine, save the `group_id` and reattach later.

```python
# Original session
batch = train.run_async_map(configs)
print(f"Batch ID: {batch.group_id}")  # e.g., "grp-a1b2c3d4"

# Later, from any machine with access to the same GCP project
batch = kinetic.attach_batch("grp-a1b2c3d4")
results = batch.results()
```

`attach_batch()` downloads the group manifest from GCS and rebuilds a
`JobHandle` for each child. It keeps the index alignment. If the
original batch had 10 inputs, and a crash stopped it after 7, the
`batch.jobs` list still has 10 entries. The 3 empty slots hold `None`.

If the manifest names fewer children than the batch expects, the
original `map()` is still at work. The handle then polls the manifest in
a background thread until the rest of the children appear, or until
`poll_timeout` ends the poll.

:::{note}
Kinetic writes a warning when the manifest of a reattached batch names
fewer children than expected. This shows a partial submission.
:::

**Parameters:**

- **`group_id`** (`str`): The batch identifier (e.g., `"grp-a1b2c3d4"`).
- **`project`** (`str | None`, default `None`): GCP project. Uses the
  default when `None`.
- **`cluster`** (`str | None`, default `None`): GKE cluster name. Uses
  the default when `None`.
- **`poll_interval`** (`float`, default `10.0`): Seconds between
  manifest polls while the batch is partially submitted.
- **`poll_timeout`** (`float | None`, default `1800.0`): Maximum seconds
  to poll for the remaining children. After the timeout, the handle
  reports the submission as complete, and the empty slots stay `None`.
  Reattach again to pick up the children that started since then.
  `None` polls forever. Use `None` only when you are sure that the
  original process is alive, because a dead submitter then blocks
  `wait()` and `results()` forever.

### Children that Kinetic cleaned up

`results(cleanup=True)` deletes the GCS artifacts of each child that
gives a result, and the `handle.json` file of the child is one of those
artifacts. The group manifest stays in place, so `attach_batch()` still
finds the batch. But it cannot rebuild a `JobHandle` for a child that it
cleaned up.

Kinetic treats such a child as terminal, and not as a child that is
still on the way. The batch reports the submission as complete, and
`wait()` and `results()` return immediately. The slot of that child
stays `None`, and `results()` gives `None` at that position.

`unavailable_children` shows which children are in this state. It maps
the child index to the job ID from the manifest.

```python
batch = kinetic.attach_batch("grp-a1b2c3d4")
print(batch.unavailable_children)
# {0: 'job-1a2b3c4d', 1: 'job-5e6f7a8b'}
```

A `None` slot that `unavailable_children` does not name is an input that
the original `map()` never submitted.

## Cleanup

There are two ways to clean up resources after a batch completes.

### Automatic cleanup via `results()`

By default, `results(cleanup=True)` deletes each child's Kubernetes
resources and GCS artifacts after downloading its result. The group
manifest is preserved, so `attach_batch()` still works.

```python
# Each child is cleaned up as its result is downloaded
results = batch.results()  # cleanup=True is the default
```

:::{important}
This cleanup deletes the result of each child. A later `attach_batch()`
cannot collect those results a second time. Use `cleanup=False` when you
want to reattach later and read the results again. See
[Children that Kinetic cleaned up](#children-that-kinetic-cleaned-up).
:::

### Full teardown

To delete everything — all children's resources and the group manifest
itself — call `cleanup()` on the handle:

```python
batch.cleanup(k8s=True, gcs=True)
```

**Parameters:**

- **`k8s`** (`bool`, default `True`): Delete Kubernetes resources
  (Jobs/pods) for each child.
- **`gcs`** (`bool`, default `True`): Delete GCS artifacts for each
  child **and** the group manifest.

:::{important}
After calling `cleanup(gcs=True)`, the batch can no longer be reattached
via `attach_batch()` because the manifest has been deleted.
:::

## How it works

### Threading model

`run_async_map()` uses a non-daemon background thread when the
submission loop must watch the jobs after it launches them. Three
settings need this:

- `max_concurrent` is set. The default is 64. The loop must wait for a
  free slot before it launches the next input.
- `retries` is above zero. The loop must see a failure before it can
  submit that input again.
- `fail_fast` and `cancel_running_on_fail` are both `True`. The loop
  must see the first failure before it can cancel the siblings.

In these cases the thread polls the active jobs, launches new jobs, and
cancels jobs. `run_async_map()` returns the `BatchHandle` immediately.

In all other cases the calling thread submits every input, and then
`run_async_map()` returns. Kinetic starts no background thread, and the
loop does not poll the jobs. A terminal status cannot change what the
loop does next, so the loop stops as soon as the last input is
submitted.

`fail_fast` on its own is such a case. A submission error still stops
the queue immediately, because the loop sees it inside the same
submission pass. But after every input is launched, a runtime failure
has nothing left for the loop to stop.

### Manifest

A JSON manifest is written to GCS before the first job is submitted. It
records the batch metadata (group ID, expected total, function name,
tags) and is updated after each successful submission with the child's
job ID and attempt count. This enables crash recovery: `attach_batch()`
reads the manifest to determine which jobs were submitted and
reconstructs the handle.

### Group ID

Each batch gets a unique identifier in the format `grp-{8-hex-chars}`
(e.g., `grp-a1b2c3d4`). This ID is set on each child `JobHandle` as
`group_id`, along with `group_kind="map"` and the child's `group_index`.

### Submission errors

A call to the function can raise, for example with a packaging error or
a validation error. Kinetic then keeps the exception, and the related
slot in `batch.jobs` stays `None`. Read these errors from
`batch.submission_failures`, which maps the input index to the
exception.

`results()` reports them too. With `return_exceptions=True`, it puts the
exception at that position in the result list. With
`return_exceptions=False`, it raises a `BatchError` that holds the same
map in `BatchError.submission_failures`. These inputs never became
jobs, so `BatchError.failures` does not list them.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em` Detached Jobs
:link: async_jobs
:link-type: doc

The single-job `run_async()` workflow each child of a batch is built on.
:::

:::{grid-item-card} {octicon}`zap;1em` Cost Optimization
:link: cost_optimization
:link-type: doc

Fan-out amplifies both throughput and spend; concurrency limits and spot
instances matter here.
:::

:::{grid-item-card} {octicon}`history;1em` Checkpointing
:link: checkpointing
:link-type: doc

Each child writes to its own `KINETIC_OUTPUT_DIR`; useful for long
per-job work inside a batch.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

What to do when children stick in `PENDING` or repeatedly fail.
:::
::::
