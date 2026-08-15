# Distributed Training

:::{admonition} Who this is for
:class: note

Users whose model or batch size has outgrown a
single TPU host. Kinetic + the [Pathways](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)
backend lets you treat a multi-host TPU slice as one logical machine,
without writing your own multi-process JAX coordination.
:::

For single-host slices (everything that fits on one TPU node like
`tpu-v5litepod-8`), you don't need this page — your existing JAX or
Keras code already uses every chip on the node.

## A first multi-host run

Pick a multi-host accelerator:

```python
import kinetic


@kinetic.run(accelerator="tpu-v6e-16")
def train_distributed():
  import jax

  print(f"Total devices across all hosts: {jax.device_count()}")
  print(f"This host: {jax.process_index()} of {jax.process_count()}")
  # ... your training code ...
```

Whether a slice is multi-host depends on the topology and the per-VM
chip count, not on the accelerator string alone. For example,
`tpu-v5litepod-2x2` (4 chips on one VM) and `tpu-v5litepod-2x4` (8 chips on one
VM via `ct5lp-hightpu-8t`) are both single-host, while `tpu-v5litepod-16`
(4×4 across four 4-chip VMs) and `tpu-v6e-16` (4×4 across four 4-chip VMs)
are multi-host. See [Accelerators](../accelerators.md) for the full
topology table.

Pathways is **auto-selected** for multi-host slices — Kinetic resolves
`backend="pathways"` whenever the accelerator's topology spans more than
one node, so the example above doesn't need to set it explicitly. You
only need to pass `backend="pathways"` yourself if you want to develop
against the Pathways code path on a single-host slice — handy for
shortening the iteration loop before you scale up.

## Data parallelism with Keras

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

For a richer end-to-end example using a real model, see
[`pathways_example.py`](../examples.md) and
[`gemma_sft_pathways_distributed.py`](../examples.md).

## How to think about it

Each host runs its own copy of your function. JAX collectives
(`jax.lax.psum`, sharding, `pmap`) and Keras's distribution APIs handle
the actual cross-host communication. Kinetic's job is to:

- Schedule the slice as a single logical job that the autoscaler treats
  atomically (no split brain).
- Run your function on every host with the right `JAX_*` env vars set.
- Stream stdout from the **leader pod** (process index 0) back to your
  local terminal. Other hosts' stdout is not aggregated; if you need it,
  fetch it directly from the per-host pods (see "Debugging distributed
  jobs" below).
- Return only the leader process's (`jax.process_index() == 0`) value
  to your local machine, so you don't get N copies of the result — and
  re-raise the failing host's exception when any host throws. The next
  section covers both.

## What comes back from which host

Every host runs the same command, so Kinetic decides ownership by
process index rather than by whoever finishes first:

- **The leader owns the result.** Only process 0 writes the job's
  `result.pkl`, so the value you get back is always the leader's. Other
  hosts discard their return value without serializing it.
- **Every host reports its own failures.** A non-leader host that raises
  writes a failure payload of its own, tagged with its process index.
- **The lowest-indexed failing host is the one that raises locally.** If
  the leader itself failed, its exception is the job's exception.
  Otherwise Kinetic re-raises the exception from the lowest-indexed host
  that recorded one, with that host's remote traceback attached and a
  note listing the other hosts that also failed.

So a worker crash no longer hides behind the leader's successful return
value, and the exception you see does not depend on which pod reached
Cloud Storage last.

:::{admonition} Return values must come from process 0
:class: tip

Only the leader's return value survives, so anything you need locally
has to be on process 0 by the time your function returns. Gather
sharded state with a JAX collective (or `jax.device_get` on a fully
replicated array) before returning it — a value computed only on
process 3 is discarded with the rest of that host's return.
:::

A host that the kubelet kills outright — OOM, spot preemption, node
eviction — never gets to write a payload. The job still fails, and
Kinetic reports the pod exit summaries it can still collect from
Kubernetes instead of a remote traceback.

:::{warning}
**When not to use this:** if your model and batch fit on a single TPU
host, stay there. Multi-host adds startup latency, requires Pathways,
and a single host failure fails the whole slice. Move to multi-host
only when you've outgrown one node.
:::

## Failure modes and recovery

Multi-host jobs fail differently from single-host jobs. The most common
ones, with what to actually do:

- **Slow startup (5–10 minutes for the first multi-host run).** A fresh
  TPU multi-host slice has to provision multiple VMs and boot Pathways.
  This is expected; don't kill the job thinking it's stuck. If startup
  consistently exceeds 10 minutes, run `kinetic init` and choose
  `troubleshoot`, and check your TPU quota.
- **Topology mismatch.** Your code's expected device count doesn't
  match `jax.device_count()` on the slice. Symptom: shape errors deep
  in `pmap` or sharding. *Fix:* compute mesh shapes from
  `jax.device_count()` and `jax.process_count()` instead of hardcoding.
- **One host hangs, the slice times out.** A single host that fails
  collective communication takes the slice with it. JAX raises a
  collective timeout on every host. *Fix:* the exception Kinetic raises
  names the host that reported it; when every host reports the same
  collective timeout, read logs from the per-host pods and look for the
  divergent one. Common causes are uneven data loading or a Python
  exception on one host before the collective.
- **Spot preemption.** Multi-host slices on spot capacity die together
  if any one host is preempted. *Fix:* don't use spot for multi-host
  unless you can absorb full restarts (and have checkpoints).
- **Quota exhaustion mid-run.** A scheduled slice can be delayed
  indefinitely if regional quota is full. Symptom: job stuck in
  `PENDING` for > 10 min on a multi-host accelerator. *Fix:* check
  Cloud Console quota for your accelerator type; consider switching
  zones.

:::{admonition} Recommended checkpoint frequency
:class: tip

For any multi-host run, write a
checkpoint at least every 10 minutes of wall time. The base rate of
preemption, quota issues, and slice-wide failures is high enough that
unbounded loss windows are not worth the throughput. See
[Checkpointing](checkpointing.md) for the API.
:::

## Debugging distributed jobs

`kinetic jobs logs <id>` (and `--follow` while the job is running)
returns the **leader pod's** stdout, which is what `print()` calls on
process index 0 produce. To gate output to that one process, guard
print statements with `jax.process_index()`:

```python
import jax

if jax.process_index() == 0:
  print(f"epoch {epoch}: loss={loss}")
```

For non-leader hosts, fetch logs directly from the per-host pods.
`kubectl get pods -n <namespace> | grep <job-id>` lists every pod in
the slice; `kubectl logs <pod-name>` then returns that host's stdout.
Cloud Logging in the GCP Console offers the same view through a UI
filter on the job name.

If a job fails on any host, Kinetic catches the exception and re-raises
it locally with that host's stack trace and process index, so you
usually do not need to inspect non-leader logs to diagnose a crash. Go
to the pod logs when the raised error is a collective timeout (the
symptom on every host, not the cause) or when the message says the pod
was terminated before it could report anything.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`cpu;1em` Accelerators
:link: ../accelerators
:link-type: doc

Slice topologies and naming.
:::

:::{grid-item-card} {octicon}`history;1em` Checkpointing
:link: checkpointing
:link-type: doc

Frequent checkpoints are essential here.
:::

:::{grid-item-card} {octicon}`server;1em` Multiple Clusters
:link: clusters
:link-type: doc

When to isolate multi-host TPUs from the rest of your workloads.
:::
::::
