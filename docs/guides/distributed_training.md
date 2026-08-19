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
- Return only the leader process's (`jax.process_index() == 0`) value to
  your local machine, so you do not get N copies of the result.
- Raise the exception of the host that failed, if a host fails. The
  next section gives the rules for the result and for the exception.

## Which host reports the result

All hosts run the same command. Kinetic uses the process index to decide
which host reports the result, and which host reports an error:

- **The leader writes the result.** Only process 0 writes the result
  file for the job. You therefore always get the return value of the
  leader. Each other host discards its own return value.
- **Each host reports its own failure.** A host that is not the leader
  writes a failure record. The record contains the process index of that
  host.
- **The failing host with the lowest index reports the error.** If the
  leader fails, Kinetic raises the exception of the leader. If the leader
  does not fail, Kinetic raises the exception of the failing host with
  the lowest index. Kinetic attaches the remote traceback of that host,
  and a note that lists each other host that also failed.

A failure on one host therefore cannot make the job look successful.
For the same failure, you always get the same local error.

:::{admonition} Return values must come from process 0
:class: tip

Kinetic keeps the return value of the leader only. Put the data that you
need on process 0 before your function returns. Use a JAX collective to
gather sharded data, or use `jax.device_get` on a fully replicated
array. Kinetic discards a value that only process 3 has.
:::

Some failures stop a pod before it can write a failure record. Examples
are an out-of-memory kill, a Spot preemption, and a node eviction. The
job still fails. In this case Kinetic has no remote traceback to show,
and reports the exit code of each pod instead.

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
  collective timeout on every host. *Fix:* the local error names the host
  that reported it. If all hosts report the same collective timeout, read
  the logs of each pod and find the host that is different. Common causes
  are uneven data loading or a Python exception on one host before the
  collective.
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

If a job fails on any host, Kinetic catches the exception and raises it
locally. The local error gives the stack trace and the process index of
that host. Usually you do not need the logs of the other pods.

Read the pod logs in two cases:

- The local error is a collective timeout. All hosts report this error,
  so it does not tell you which host is at fault.
- The local error tells you that Kubernetes stopped a pod before the pod
  could report a failure.

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
