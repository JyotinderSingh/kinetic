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
- **`return_exceptions`** (`bool`, default `False`): When `True`, failed
  positions contain the exception object instead of raising
  `BatchError`. When `False`, any failure raises `BatchError`. A job
  that fails and an input that fails at submission time both count as a
  failure.

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
`results()` raises a `BatchError`. An input that fails at submission
time raises a `BatchError` too.

```python
try:
  results = batch.results(cleanup=False)
except kinetic.BatchError as e:
  print(e)  # "Batch grp-a1b2c3d4: 2 of 8 jobs failed"
  for job in e.failures:
    print(f"{job.job_id}: {job.status().value}")
    print(job.tail(n=20))
  for idx, exc in e.submission_failures.items():
    print(f"Input {idx} failed at submission: {exc}")
  # e.partial_results holds the result at each successful position
  # and None at each failed position.
```

`BatchError` provides four attributes:

* **`group_id`**: The batch identifier.
* **`failures`**: A list of `JobHandle` objects for the jobs that started and then failed. The list holds only `JobHandle` objects, so `job.job_id` and `job.status()` are always safe to call.
* **`submission_failures`**: A dict mapping input indices to the exceptions from submission. These inputs never became jobs, have no `JobHandle`, and never appear in `failures`.
* **`partial_results`**: With `ordered=True`, a list aligned with `inputs`, where a successful position holds the result and a failed position holds `None`. With `ordered=False`, a shorter list in completion order holding only the successful results.

The example passes `cleanup=False`. With the default `cleanup=True`, `results()` deletes the Kubernetes Job of every child before it raises `BatchError`. After that deletion, `job.status()` returns `NOT_FOUND` and `job.tail()` raises `RuntimeError`, because the pod is gone. Even with `cleanup=False`, Kubernetes deletes a finished Job about 10 minutes after it ends, so read the logs soon after the failure. This retention window applies to single-host jobs on the GKE backend. A multi-host Pathways job has no retention window; its resources stay until a cleanup call deletes them.

### Tolerate failures

Pass `return_exceptions=True` to collect the results without raising an exception. A failed position holds the exception object.

```python
results = batch.results(return_exceptions=True)
for i, r in enumerate(results):
  if isinstance(r, Exception):
    print(f"Job {i} failed: {r}")
  else:
    print(f"Job {i}: {r}")
```

### Inspect failed jobs

`failures()` returns the handles of the jobs with status `FAILED`. It excludes `NOT_FOUND`, because that status is ambiguous. A job can be `NOT_FOUND` because Kinetic cleaned up its Kubernetes resources, not because the job failed. Use `statuses()` for a finer inspection.

After `results()` has run, `failures()` returns the failures from that collection pass, and not the live status of each job. This keeps the list correct after `cleanup=True` deletes the Kubernetes resources.

```python
batch.wait()
for job in batch.failures():
  print(f"{job.job_id}: {job.tail(n=20)}")
```

`job.tail()` reads the pod log, so call it while the pod exists. Call it after `wait()` and before `results()`, or after `results(cleanup=False)`, within the 10-minute Kubernetes retention window.

### Submission failures

The call that submits an input can raise, for example because of a packaging or validation error. Kinetic then records the exception and leaves `batch.jobs[idx]` as `None`. Kinetic does not retry a submission failure.

`failures()` reports only the jobs that started. To inspect inputs that failed before they became jobs, read `batch.submission_failures`. The `batch.submission_failures` property returns a dict mapping the input index to the exception. `results()` reports these inputs as failures, but they have no `JobHandle` and do not appear in `failures()`. The position in the results holds the exception object only when `return_exceptions=True`. `wait()` logs a warning when a batch has submission failures.

```python
for index, exc in batch.submission_failures.items():
  print(f"input {index} failed to submit: {exc}")
```

## Retries

The `retries` argument sets the number of additional attempts that an input gets after a job failure. The total number of attempts per input is `1 + retries`.

```python
batch = train.run_async_map(configs, retries=2)
# Each input gets up to 3 attempts (1 initial + 2 retries)
```

* Kinetic starts a retry when a job reaches `FAILED` or `NOT_FOUND`.
* Before each retry, Kinetic deletes the Kubernetes resources of the previous attempt and keeps its Cloud Storage (GCS) artifacts for debugging.
* Each attempt is a new job with a new job ID. The group manifest tracks the attempt count per job, which allows `attach_batch()` to distinguish retries from initial submissions.
* Kinetic does not retry a submission failure, such as an error raised during function packaging or validation, because repeated attempts would fail identically.
* Kinetic does not retry a cancelled job. `cancel()` marks child jobs so that the resulting `NOT_FOUND` status never triggers a new attempt.

:::{note}
When `retries > 0`, Kinetic runs the submission loop in a background thread, so that it can poll for failures and resubmit.
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

* **Default `64`:** Kinetic starts a new job each time an active job ends.
* **`None`:** Kinetic submits all inputs immediately, with no concurrency limit. With `retries=0` (the default) and when `fail_fast` and `cancel_running_on_fail` are not both `True`, the submission runs in the calling thread before `run_async_map()` returns. See [Threading model](https://www.google.com/search?q=%23threading-model).
* The value must be a positive integer when set. Passing `0` or a negative value raises `ValueError`.

In every case `run_async_map()` returns the `BatchHandle` as soon as the
submission work is handed off or complete. It never waits for the jobs
to finish. Use `wait()` or `results()` when you want to block.

:::{note}
Kinetic logs a warning when you submit more than 100 inputs with
`max_concurrent=None`. Set a limit to control the resource usage.
:::
## Cancellation and fail-fast

### Fail-fast behavior

The `fail_fast` and `cancel_running_on_fail` arguments control what happens when a job fails.

| `fail_fast` | `cancel_running_on_fail` | On the first failure |
| --- | --- | --- |
| `False` (default) | `False` (default) | All remaining jobs continue. Kinetic reports the failures at the end. |
| `True` | `False` | Kinetic starts no new jobs. Jobs that already run continue to the end. |
| `True` | `True` | Kinetic starts no new jobs and cancels all running jobs at once. |
| `False` | `True` | **No effect.** `cancel_running_on_fail` applies only when `fail_fast=True`. |

```python
# Stop the batch as soon as any job fails, and cancel all running jobs
batch = train.run_async_map(
  configs,
  fail_fast=True,
  cancel_running_on_fail=True,
)
```

A "failure" here is either a submission failure (when the call raises) or a runtime failure: a remote job reaching `FAILED` or `NOT_FOUND` status after all of its attempts.

:::{note}
`run_async_map(max_concurrent=None, retries=0, fail_fast=True)` does not return at once. In that configuration the submission loop runs in the calling thread, and `fail_fast` makes the loop poll until every job is terminal. If you want the call to return at once, set a concurrency limit or leave `fail_fast=False`.
:::

### Manual cancellation

`batch.cancel()` cancels every submitted job that is not terminal. It stops the full collection at any time and is independent of the `fail_fast` setting.

```python
batch.cancel()
```

`batch.cancel()` performs three actions:

* It deletes the Kubernetes resource of each job that is not terminal, while preserving Cloud Storage (GCS) artifacts for debugging.
* It drops remaining inputs held in the `max_concurrent` queue so Kinetic does not launch them.
* It marks child jobs as cancelled so Kinetic does not retry them, even when `retries > 0`.

A cancelled job reports the status `NOT_FOUND` because its Kubernetes resource has been deleted. `wait()` returns after every job that started becomes terminal, and the slot of an input that never launched remains `None`.

Because a cancelled job produces no result, `results()` raises a `BatchError` that lists those jobs in `failures`. Pass `return_exceptions=True` to `results()` to collect results from any jobs that finished before cancellation.

:::{note}
The subsequent trailing bullet points describing workaround configurations like `max_concurrent=None` or `fail_fast=True` are obsolete and should be removed, as `cancel()` now natively stops queued and retried jobs.
:::

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
ends. After the timeout, the handle reports the submission as complete,
and the empty slots stay `None`.

:::{caution}
If the original process crashed during submission, missing children
never appear. With `poll_timeout=None`, `wait()`, `results()`, or
`as_completed()` will block indefinitely. Always pass an explicit
`poll_timeout` when you reattach. In the original session, use
`results(cleanup=False)` if you plan to reattach later.
:::

**Parameters:**

* **`group_id`** (`str`): The batch identifier (for example
`"grp-a1b2c3d4"`).
* **`project`** (`str | None`, default `None`): A one-off override. The
active profile supplies the project when `None`.
* **`cluster`** (`str | None`, default `None`): A one-off override. The
active profile supplies the cluster when `None`.
* **`poll_interval`** (`float`, default `10.0`): The number of seconds
between manifest polls when children are missing.
* **`poll_timeout`** (`float | None`, default `1800.0`): The maximum
number of seconds to poll for missing children. After the timeout, the
handle reports the submission as complete, and the empty slots stay
`None`. Reattach again to pick up children that started since then.
`None` polls forever. Use `None` only when you are sure that the
original process is still running.

### Children that Kinetic cleaned up

`results(cleanup=True)` deletes the Cloud Storage artifacts of each child
that yields a result, including the child's `handle.json` file. The group
manifest stays in place, so `attach_batch()` still finds the batch, but it
cannot rebuild a `JobHandle` for a cleaned-up child.

Kinetic treats such a child as terminal rather than in-flight. The batch
reports the submission as complete, and `wait()` and `results()` return
immediately. The slot for that child stays `None`, and `results()` returns
`None` at that position.

The `unavailable_children` property shows which children are in this state.
It maps the child index to the job ID from the manifest:

```python
batch = kinetic.attach_batch("grp-a1b2c3d4")
print(batch.unavailable_children)
# {0: 'job-1a2b3c4d', 1: 'job-5e6f7a8b'}
```

A `None` slot that does not appear in `unavailable_children` represents an
input that the original `map()` never submitted.

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

:::{important}
This cleanup deletes the result of each child. A later `attach_batch()`
cannot collect those results a second time. Use `cleanup=False` when you
want to reattach later and read the results again. See
[Children that Kinetic cleaned up](#children-that-kinetic-cleaned-up).
:::

### Full teardown

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

`run_async_map()` uses a non-daemon background thread when the submission loop must monitor jobs after launching them. Three settings require this:

* **`max_concurrent` is set** (the default is 64): The loop must wait for a free slot before launching the next input.
* **`retries > 0`**: The loop must detect a failure before resubmitting that input.
* **`fail_fast` and `cancel_running_on_fail` are both `True**`: The loop must detect the first failure to cancel sibling jobs.

In these cases, the thread polls active jobs, starts new jobs, and cancels running jobs as needed. `run_async_map()` returns the `BatchHandle` immediately. Because the background thread is not a daemon thread, the Python process stays alive until the submission ends.

In all other cases, Kinetic submits every input in the calling thread and starts no background thread. The loop does not poll jobs, and `run_async_map()` returns as soon as the last input is submitted.

`fail_fast` on its own falls into this category: a submission error stops the queue immediately because the loop catches it during submission, but once every input is launched, runtime failures have no remaining effect on the submission loop.

### Manifest

Kinetic writes a JSON manifest to `gs://{jobs bucket}/_groups/{group_id}/manifest.json`
before it submits the first job. The manifest records the batch metadata
(group ID, expected total, function name, name, and tags). Kinetic
updates the manifest after each successful submission with the child's
index and job ID. `attach_batch()` reads the manifest to find the
submitted jobs and rebuilds the handle from each child's `handle.json`.

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
