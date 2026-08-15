# Distributed Training

This page covers jobs on TPU slices that span more than one host. Read
this page if your model or your batch no longer fits on one TPU host.
A multi-host job runs one pod per host, and Kinetic runs it on the
`pathways` backend. This page explains what changes: how Kinetic
starts the job, which log you see, what the return value is, and how a
multi-host job fails.

If your slice fits on one host, for example `tpu-v5litepod-8`, you do
not need this page. Your JAX or Keras code already uses every chip on
that host.

## Single-host or multi-host?

The **Hosts** column of the TPU table on the [Accelerators](../accelerators.md#tpus)
page decides. A slice with one host is single-host. A slice with more
than one host is multi-host. Some examples:

| Accelerator | Topology | Hosts | Backend |
| ----------- | -------- | ----- | ------- |
| `tpu-v5litepod-8` | 2x4, one 8-chip VM | 1 | `gke` |
| `tpu-v6e-8` | 2x4, two 4-chip VMs | 2 | `pathways` |
| `tpu-v5litepod-16` | 4x4, four 4-chip VMs | 4 | `pathways` |
| `tpu-v6e-16` | 4x4, four 4-chip VMs | 4 | `pathways` |
| `tpu-v5p-8` | 2x2x2, two 4-chip VMs | 2 | `pathways` |

The chip count alone does not tell you the host count. `tpu-v5litepod-8`
and `tpu-v6e-8` both have 8 chips, but the v5e slice is one 8-chip VM
and the v6e slice is two 4-chip VMs. Every v6e slice and every v5p
slice is multi-host. Run `kinetic accelerators` to print the names, chip
counts, and topologies that your installed version accepts. That command
does not print the host count, so use the table on the Accelerators page.

Kinetic selects the backend from the host count. If the slice has more
than one host, Kinetic selects `backend="pathways"`. You do not set
`backend` yourself.

## Before you start

- The cluster needs a node pool for the multi-host accelerator, for
  example `kinetic pool add --accelerator tpu-v6e-16`. See
  [Clusters and Node Pools](clusters.md).
- The cluster needs the LeaderWorkerSet controller. `kinetic up`
  installs it. If a submit fails with `LeaderWorkerSet CRD not found`,
  run `kinetic up` on that cluster.

## A first multi-host run

This example selects a multi-host accelerator and prints the process
layout:

```python
import kinetic


@kinetic.run(accelerator="tpu-v6e-16")
def train_distributed():
  import jax

  print(f"Total devices across all hosts: {jax.device_count()}")
  print(f"This host: {jax.process_index()} of {jax.process_count()}")
  # ... your training code ...
  return jax.device_count()


print(train_distributed())
```

On `tpu-v6e-16`, `jax.process_count()` is 4, `jax.local_device_count()`
is 4, and `jax.device_count()` is 16. Your terminal shows the log of
the leader host only, so you see one `This host: 0 of 4` line.

## How a multi-host job runs

Kinetic creates one Kubernetes LeaderWorkerSet for the job. The set has
one leader pod and one worker pod for each other host. Every pod runs
the same command, downloads the same artifacts, and calls your function.
Kinetic sets these environment variables on every pod:
`MEGASCALE_COORDINATOR_ADDRESS`, `MEGASCALE_NUM_SLICES`,
`TPU_WORKER_ID`, `JAX_PLATFORMS`, and `KERAS_BACKEND=jax`. Each pod
runs one JAX process, and `jax.process_count()` equals the host count.

Your code does the cross-host communication. JAX collectives
(`jax.lax.psum`, sharding, `pmap`) and the Keras distribution API
handle it. Kinetic and GKE do these things:

- The LeaderWorkerSet creates one pod per host, and the GKE autoscaler
  starts every VM of the slice for the job.
- Kinetic streams the log of the **leader pod** to your terminal.
  Kinetic does not stream or interleave the logs of the other pods. See
  [Debugging distributed jobs](#debugging-distributed-jobs) for how to
  read them.
- Kinetic treats a failure on any host as a failure of the job.

:::{note}
The name `pathways` refers to the Kinetic backend for multi-host TPU
jobs. Kinetic does not deploy Pathways-on-Cloud proxy servers or a
resource manager. Each host runs its own JAX process, and
`jax.process_count()` is greater than 1.
:::

:::{warning}
If your model and your batch fit on one host, stay on one host. A
multi-host job starts more slowly, needs the LeaderWorkerSet
controller, and fails as a whole if one host fails. Move to
multi-host only if one host is not enough.
:::

## The return value

Every host uploads its return value to the same `result.pkl` object in
the jobs bucket. Kinetic does not select the value of process 0. The
value that Kinetic returns to you is the value that the last host wrote.

Write your function so that this rule does not matter:

- Return the same small value from every host, for example a metric that
  every host computes.
- Or return a value from process 0 only and return `None` from the other
  hosts. Then check the returned value on the client, because the client
  can receive the `None`.

```python
@kinetic.run(accelerator="tpu-v6e-16")
def train():
  import jax

  final_loss = ...  # every host holds the same value after the collective
  return float(final_loss)  # same value on every host
```

Write large outputs, such as model weights and checkpoints, under
`KINETIC_OUTPUT_DIR` from process 0. See
[Outputs and Checkpoints](checkpointing.md).

## Failures and the exception that you see

A failure on any host fails the job. What you see on the client depends
on which pod wrote `result.pkl` last:

- If the failed pod wrote last, Kinetic raises the remote exception on
  the client with the traceback of that host attached.
- If the leader finished and uploaded a success payload, but a worker
  pod failed, Kinetic does not return the leader value. Kinetic raises a
  `RuntimeError`. The message names the failed pods, their exit codes,
  and the last 30 log lines of each failed pod. Kinetic keeps the job
  artifacts in the jobs bucket for inspection.

In both cases, a blocking call and `JobHandle.result()` with the default
`cleanup` then delete the LeaderWorkerSet and its pods. Read the pod
logs from Cloud Logging after that point.

For a detached job, `JobHandle.status()` reports `FAILED` if any
worker pod failed, also if the leader pod finished. See
[Detached Jobs](async_jobs.md).

## Test the multi-host code path on one host

Set `backend="pathways"` on a single-host accelerator to run the same
LeaderWorkerSet code path with one pod. Use this run to test a script
before you request a large slice.

```python
@kinetic.run(accelerator="tpu-v5litepod-8", backend="pathways")
def smoke_test():
  import jax

  return jax.process_count()  # 1 on a single-host slice
```

On one host, `jax.process_count()` is 1, so this run does not test the
cross-host collectives. The run tests the job flow, the image, and your
code up to the first collective.

## Data parallelism with Keras

The Keras distribution API sees every device on every host. Build a
device mesh over all devices and set a `DataParallel` distribution:

```python
@kinetic.run(accelerator="tpu-v6e-16")
def train_data_parallel():
  import keras

  devices = keras.distribution.list_devices()
  device_mesh = keras.distribution.DeviceMesh(
    shape=(len(devices),),
    axis_names=["batch"],
    devices=devices,
  )
  keras.distribution.set_distribution(
    keras.distribution.DataParallel(device_mesh=device_mesh)
  )

  model = keras.Sequential([...])
  model.compile(...)
  model.fit(...)
```

Two complete scripts show multi-host runs and `pathways` backend runs:

- [`pathways_example.py`](../examples/pathways_example.md) runs on
  `tpu-v6e-16`. The script prints the process layout, checks a
  cross-host `psum`, and trains a small Keras model.
- [`gemma_sft_pathways_distributed.py`](../examples/gemma_sft_pathways_distributed.md)
  uses the `DataParallel` pattern above with a Gemma model. The script
  runs on the single-host slice `tpu-v5litepod-2x4` with
  `backend="pathways"`. Change the accelerator to a multi-host slice to
  run the same code on many hosts.

## Failure modes and recovery

Multi-host jobs fail in ways that single-host jobs do not. The most
common ones, with the action for each:

- **Slow startup.** The autoscaler must start every VM of the slice, and
  Cloud Build runs if the image is new. Do not stop the job because it
  shows no progress. Run `kinetic jobs status <job-id>` in a second
  terminal to see the current status. If the job stays `PENDING` for
  more than 10 minutes, check the accelerator quota of your project. See
  [Troubleshooting](../troubleshooting.md#scheduling-and-quota-issues).
- **Topology mismatch.** Your code expects a device count that does not
  match `jax.device_count()` on the slice. Symptom: shape errors inside
  `pmap` or sharding. *Fix:* compute mesh shapes from
  `jax.device_count()` and `jax.process_count()`. Do not hard-code them.
- **One host hangs, the slice waits.** A host that does not reach a
  collective stops the whole slice. The other hosts wait at the
  collective, and the job makes no progress. *Fix:* read the log of
  every pod with `kubectl` (see below) and find the host that diverged.
  Common causes are uneven data loading and a Python exception on one
  host before the collective.
- **Spot preemption.** If Google Cloud preempts one host of a Spot slice,
  the whole slice loses its state. The LeaderWorkerSet restart policy is
  `RecreateGroupOnPodRestart`. The controller recreates every pod of the
  group, and your function starts again from the beginning on every
  host. *Fix:* use on-demand capacity for multi-host jobs, or make sure
  that the job writes checkpoints and can resume from them.
- **No capacity or no quota.** A multi-host slice needs several VMs of
  one type in one zone. The job stays `PENDING` if the zone has no
  capacity or your project has no quota. *Fix:* check the quota page in
  the Cloud Console for the accelerator type, use a
  [capacity reservation](reservations.md), or use a cluster in another
  zone.
- **A second job on the same pool waits.** By default, a multi-host node
  pool scales up to one slice. A second job on the same accelerator
  waits until the first job ends. *Fix:* add a second pool for the same
  accelerator. For warm capacity, add a pool with `--min-nodes` set to a
  multiple of the host count.

:::{admonition} Write checkpoints often
:class: tip

Write checkpoints at short intervals in a multi-host run, for example
every 10 minutes of wall time. Preemption, quota problems, and
slice-wide failures are frequent enough that a long gap between
checkpoints costs more than the checkpoint itself. The default output
directory is per job, so pass an explicit `output_dir=` if you resume
a run from an earlier job. See
[Outputs and Checkpoints](checkpointing.md).
:::

## Debugging distributed jobs

`kinetic jobs logs <job-id>` returns the log of the **leader pod**. Add
`--follow` while the job runs, or `--tail N` for the last `N` lines.
Every host runs your `print()` calls. Guard the calls that you want to
see one time with `jax.process_index()`:

```python
import jax

if jax.process_index() == 0:
  print(f"epoch {epoch}: loss={loss}")
```

Kinetic does not stream the logs of the other hosts. Read them with
`kubectl`. Every pod of the job carries the label `job-id=<job-id>`:

```bash
kubectl get pods -n <namespace> -l job-id=<job-id>
kubectl logs -n <namespace> <pod-name>
```

`<namespace>` is the namespace of the active profile, `default` unless
you changed it. The leader pod is `keras-pathways-<job-id>-0`. The
worker pods have the same prefix and a further index suffix.

A blocking call and `JobHandle.result()` with the default `cleanup`
delete the pods after the job ends. Read the pod logs while the job
runs, or use a detached job and read the logs before you call
`result()`. Cloud Logging in the Cloud Console keeps the same logs after
Kinetic deletes the pods. Filter on the pod name.

If a pod fails, the failure message on the client already contains the
last 30 log lines of each failed pod. Read those lines first. Read the
full pod logs if those lines are not enough.

`debug=True` attaches a debugger to the leader pod. Kinetic holds the
worker pods until the leader is ready. See
[Interactive Debugging](debugging.md#multi-host-debugging).

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`cpu;1em` Accelerators
:link: ../accelerators
:link-type: doc

The TPU table with the Hosts column that decides the backend.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: checkpointing
:link-type: doc

`KINETIC_OUTPUT_DIR` and resumable training for long runs.
:::

:::{grid-item-card} {octicon}`server;1em` Clusters and Node Pools
:link: clusters
:link-type: doc

Add a node pool for a multi-host slice, or isolate it in its own cluster.
:::

:::{grid-item-card} {octicon}`clock;1em` Detached Jobs
:link: async_jobs
:link-type: doc

Submit a long multi-host job and collect the result later.
:::
::::
