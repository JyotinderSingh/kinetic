# Cost Optimization

This page explains what a Kinetic cluster costs while it is idle and
while a job runs, and how to keep the bill low. Read this page after
your [first run](../getting_started.md), when you decide how long the
cluster stays up and which node pools it holds. Every command below
reads the project, the zone, and the cluster from the active profile.

## What you pay for

A Kinetic cluster has two kinds of cost: the cost of a running job and
the cost of an idle cluster.

**While a job runs**, you pay for the accelerator node that runs the pod.
The cluster autoscaler starts the node when the job needs it and deletes
the node after an idle period. On a Spot node pool, the node costs less
(see [Spot VMs](#spot-vms)).

**While the cluster exists**, you pay for these resources even when no
job runs:

- The GKE control plane, about $0.10 per hour per cluster. Google Cloud
  gives each billing account a
  [monthly credit](https://cloud.google.com/kubernetes-engine/pricing)
  that covers the control plane of one zonal cluster.
- One `e2-standard-4` node in the default node pool. `kinetic up` creates
  that node, and the node runs at all times. The node runs the cluster
  system pods and `accelerator="cpu"` jobs. The free credit does not
  cover the node.
- One Cloud NAT gateway. The cluster nodes have private IP addresses,
  and the gateway gives them internet access. Google Cloud bills the
  gateway separately.
- Storage in the jobs bucket and the builds bucket. Both buckets delete
  objects that are older than 30 days.

Cloud Build bills each image build. Kinetic builds an image only when
your dependency set changes (see
[Container image builds](#container-image-builds)).

`kinetic down` deletes all of these resources and stops all of these
costs. See [Delete the cluster](#delete-the-cluster).

## Scale to zero

By default, every accelerator node pool scales to zero. A job goes
through these steps:

:::{container} kinetic-steps
1. **Submit.** You call a function that has the `@kinetic.run()`
   decorator. Kinetic creates a Kubernetes Job on the cluster.
2. **Scale up.** The cluster autoscaler sees the pending pod and starts a
   node in the matching node pool. This step takes about 2 to 5 minutes.
3. **Run and exit.** The pod runs your function and exits, with a result
   or with an exception.
4. **Scale down.** If no new job arrives within the idle window, the
   autoscaler deletes the node. The idle window is at most about 10
   minutes. The accelerator cost returns to zero.
:::

Kinetic creates the cluster with the `OPTIMIZE_UTILIZATION` GKE
autoscaling mode. In this mode, GKE removes idle nodes faster than in
the default balanced mode.

## Warm nodes with `--min-nodes`

When you add a node pool, you choose between start latency and idle
cost.

**Default: scale to zero**

```bash
kinetic pool add --accelerator gpu-l4
```

- Benefit: no accelerator cost when no job runs.
- Trade-off: after an idle period, each job waits about 2 to 5 minutes
  for a node. The wait covers the VM start and the image pull.

**Warm nodes**

If you iterate on a script and submit a job every few minutes, the wait
for a node interrupts your work. Keep one node warm with `--min-nodes`:

```bash
kinetic pool add --accelerator gpu-l4 --min-nodes 1
```

- Benefit: a job starts as soon as Kinetic schedules the pod. The node
  is up, and after the first job the node holds the image in its local
  cache.
- Trade-off: you pay for that node at all times, also at night and on
  weekends.

`kinetic up --min-nodes N` applies the same setting to the first node
pool of a new cluster.

:::{note}
Kinetic has no command to change `--min-nodes` on an existing pool. To
return a pool to scale-to-zero, remove the pool and add it again:

```bash
kinetic pool list
kinetic pool remove gpu-l4-1a2b
kinetic pool add --accelerator gpu-l4
```

The new pool gets a new generated name.
:::

For a multi-host TPU pool, `--min-nodes` counts VMs, not slices, and the
value must be a multiple of the number of hosts in the slice. For
example, a `tpu-v5litepod-16` slice has 4 hosts (see the **Hosts**
column in [Accelerators](../accelerators.md#tpus)), so `--min-nodes 4`
keeps one warm slice.

## Container image builds

Kinetic runs the pod in a container image that it builds with Cloud
Build. Cloud Build bills the build minutes, and a cold build takes about
5 to 10 minutes. You do not need to change anything to avoid repeated
builds:

- Kinetic tags each image with a hash of the dependency set and caches
  the image in the Artifact Registry repository of the cluster.
- A change to your code does not cause a build. Kinetic ships your code
  in a separate archive for every job.
- Kinetic builds a new image only when one of these inputs changes: the
  dependency file (`requirements.txt` or `pyproject.toml`), your Python
  minor version, the Kinetic version, or the accelerator category (CPU,
  GPU, or TPU).

Two habits keep the number of builds low:

- Keep the dependency file stable. Add a package one time, not once per
  experiment.
- Use one Python minor version on your machine for all jobs on a
  cluster.

The other image modes do not lower the build cost. See
[Container Images](containers.md).

## Spot VMs

Spot VMs use spare Google Cloud capacity at a large discount, up to 91%
below the on-demand price (see the
[Spot VM documentation](https://cloud.google.com/compute/docs/instances/spot)).
Google Cloud can preempt a Spot VM at any time, with 30 seconds of
notice.

Spot use in Kinetic has two parts. Both parts are necessary.

1. **The pool.** Add a node pool with `--spot`:

   ```bash
   kinetic pool add --accelerator gpu-a100 --spot
   ```

2. **The job.** Set `spot=True` on the decorator, or add the `:spot`
   suffix to the accelerator string:

   ```python
   @kinetic.run(accelerator="gpu-a100", spot=True)
   def train():
     ...
   ```

   ```python
   @kinetic.run(accelerator="gpu-a100:spot")
   def train():
     ...
   ```

The job side adds the Spot node selector and the Spot toleration to the
pod. Without them, the pod cannot schedule on the Spot pool, and the
job stays `PENDING`. A job with `spot=True` needs a `--spot` pool, and a
job without `spot=True` needs an on-demand pool. If you run both kinds
of job on one accelerator type, add two pools.

Follow these rules when you use Spot:

- **Run only fault-tolerant jobs on Spot.** Kinetic does not submit a
  preempted single-host job again. The job ends as `FAILED`, and you
  submit it again yourself. Do not use Spot for a job with a deadline or
  for a job that must run without interruption.
- **Write checkpoints under `KINETIC_OUTPUT_DIR`**, for example with
  Orbax. Pass a fixed `output_dir=` to the decorator so that the second
  submission finds the checkpoints of the first one. See
  [Resume a job from a checkpoint](checkpointing.md#resume-a-job-from-a-checkpoint).
- **Prefer single-host accelerators.** A multi-host TPU slice loses all
  of its work when Google Cloud preempts any one of its hosts. Multi-host
  slices include `tpu-v6e-8` and `tpu-v6e-16`, every `tpu-v5p` size,
  `tpu-v5litepod-16` and larger, `tpu-v4-8` and larger, and `tpu-v3-16`
  and larger. Use Spot for single-host jobs, for example on
  `tpu-v5litepod-4`, `tpu-v5litepod-8`, or `gpu-l4`.
- **Do not combine `debug=True` with `spot=True`.** A preemption ends the
  debug session. Kinetic prints a warning for this combination.

## Capacity reservations

A capacity reservation does not lower the price. It guarantees that a
node is available when the pool scales up. A reservation is useful for
accelerators in short supply, for example H100 or TPU v6e. Bind a
reservation to a pool with `--reservation`:

```bash
kinetic pool add --accelerator gpu-h100 --reservation my-h100-reservation
```

You cannot combine `--reservation` with `--spot`. Google Cloud bills a
reservation while it exists, also when no VM uses it. See
[Capacity Reservations](reservations.md).

## Node pool hygiene

Each node pool with `--min-nodes 0` costs nothing while it is idle, but
an unused pool with warm nodes costs money every hour. Review the pools
of the cluster regularly:

```bash
kinetic pool list
```

Remove each pool that you no longer use:

```bash
kinetic pool remove tpu-v5litepod-1a2b
```

## Delete the cluster

If you do not use Kinetic for days or weeks, delete the cluster:

```bash
kinetic down
```

`kinetic down` deletes the cluster, the node pools, the default node,
the NAT gateway, the Artifact Registry repository, and the jobs bucket
and the builds bucket. After that, the cluster costs nothing. The state
bucket of the project (`gs://{project}-kinetic-state`) stays, but it
holds only small state files.

:::{warning}
`kinetic down` deletes the buckets with their contents, including
everything under `KINETIC_OUTPUT_DIR`. Copy the outputs that you want to
keep to another bucket first.
:::

`kinetic up` creates a new cluster in about 5 to 10 minutes. The first
job on the new cluster builds a new image, because the image cache is
part of the cluster. See [Clusters and Node Pools](clusters.md).

## Checklist

1. Leave `--min-nodes` at `0` unless you submit a job every few minutes.
2. Keep the dependency file stable so that Kinetic reuses the cached
   image.
3. For a fault-tolerant single-host job, add a `--spot` pool and set
   `spot=True` on the job. Write checkpoints under `KINETIC_OUTPUT_DIR`.
4. Run `kinetic pool list` and remove the pools that you no longer use.
5. Run `kinetic down` when you do not need the cluster for days or
   weeks. Copy your outputs first.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`server;1em` Clusters and Node Pools
:link: clusters
:link-type: doc

What `kinetic up` creates, node pool options, and when a second
cluster is justified.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: checkpointing
:link-type: doc

Write checkpoints under `KINETIC_OUTPUT_DIR` and resume a job after a
preemption.
:::

:::{grid-item-card} {octicon}`checklist;1em` Capacity Reservations
:link: reservations
:link-type: doc

Guaranteed capacity for accelerators in short supply.
:::

:::{grid-item-card} {octicon}`package;1em` Container Images
:link: containers
:link-type: doc

How the image cache works, and when prebuilt or custom images make
sense.
:::
::::
