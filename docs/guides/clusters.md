# Clusters and Node Pools

A **cluster** is the GKE cluster, with its buckets and its image
repository, that runs your jobs. A **node pool** is a group of VMs of one
accelerator type inside a cluster. This page explains what `kinetic up`
creates and how to add and remove node pools. It also explains how a
team shares one cluster, when to run more than one cluster, and how to
delete a cluster.

## What `kinetic up` creates

`kinetic init` runs `kinetic up` on the **Create** path. You can also run
`kinetic up` yourself. One run creates one cluster and everything the
cluster needs:

| Resource | Name | Purpose |
| -------- | ---- | ------- |
| GKE cluster | `{cluster}` (default `kinetic-cluster`) | Runs the job pods. |
| Node pool | `gpu-{type}-{4 hex}` or `tpu-{type}-{4 hex}`, for example `tpu-v5litepod-1a2b` | One accelerator type. `{type}` is the type name without the count, such as `l4` or `v5litepod`. `up` adds one pool of your choice; `kinetic pool add` adds more. |
| Artifact Registry repository | `kn-{cluster}` | Holds the container images that Kinetic builds. |
| Jobs bucket | `gs://{project}-kn-{cluster}-jobs` | Job artifacts, results, `KINETIC_OUTPUT_DIR`, and the data cache. Objects expire after 30 days. |
| Builds bucket | `gs://{project}-kn-{cluster}-builds` | Cloud Build sources. Objects expire after 30 days. |
| Service accounts | `kn-{cluster}-nodes`, and one for builds | Give the pods and Cloud Build access to the buckets and the repository. |
| Cloud NAT gateway | `kn-{cluster}-nat` | Gives the private cluster nodes access to the internet. |
| State bucket | `gs://{project}-kinetic-state` | One per project. Holds the infrastructure state for all clusters in the project. |

`kinetic up` also enables the required Google Cloud APIs, installs the
LeaderWorkerSet controller and the GPU driver installer on the cluster,
configures `kubectl`, and saves a profile for the cluster. The profile
becomes active.

Useful flags:

- `--cluster NAME` and `--zone ZONE` select the cluster name and zone.
- `--accelerator SPEC` selects the first node pool without a prompt,
  for example `--accelerator tpu-v5litepod-4`. Use `cpu` for a cluster
  without an accelerator pool.
- `--min-nodes N` keeps N nodes of the first pool warm. The default is
  `0`.
- `--preview` shows the changes without applying them.
- `--yes` skips the confirmation prompt.

A second `kinetic up` for the same cluster is safe. It keeps the existing
node pools and ignores `--accelerator`. Use `kinetic pool add` and
`kinetic pool remove` to change the pools.

## Check the cluster

```bash
kinetic status            # cluster, buckets, repository, node pools
kinetic pool list         # node pools only
kinetic accelerators      # every accelerator name that Kinetic knows
kinetic accelerators --live   # marks the accelerators that have a pool
```

## Node pools

A job runs only on a node pool with the same accelerator type and, for
TPUs, the same topology. Add one pool for each accelerator that you use:

```bash
kinetic pool add --accelerator tpu-v5litepod-4
kinetic pool add --accelerator gpu-l4
kinetic pool add --accelerator gpu-a100x4 --spot
kinetic pool add --accelerator tpu-v6e-16 --reservation my-v6e-reservation
```

`kinetic pool add` accepts the same accelerator names as
`@kinetic.run(accelerator=...)`. See [Accelerators](../accelerators.md).
The command prints the generated pool name.

Options:

- `--min-nodes N` keeps N nodes running at all times. The default, `0`,
  scales the pool to zero when no job runs. Nodes cost money while they
  run, even without a job. See [Cost Optimization](cost_optimization.md).
- `--spot` uses Spot VMs. Spot VMs cost less, but Google Cloud can
  preempt them with 30 seconds of notice. A job runs on a Spot pool only
  when the decorator also sets `spot=True`. See
  [Cost Optimization](cost_optimization.md).
- `--reservation NAME` binds a capacity reservation to the pool. You
  cannot combine `--reservation` with `--spot`. See
  [Capacity Reservations](reservations.md).
- `--preview` shows the change without applying it.

Remove a pool by name:

```bash
kinetic pool list
kinetic pool remove tpu-v5litepod-1a2b
```

Two limits apply to every pool that `kinetic pool add` creates:

- A pool scales up to a fixed maximum: `--min-nodes` plus 10 nodes for a
  GPU pool, or `--min-nodes` plus the hosts of one slice for a TPU pool.
  One TPU pool therefore runs one slice at a time.
- Each node has a maximum run duration of 24 hours, except a node in a
  Spot TPU pool. GKE recycles the node after 24 hours, and a job on that
  node fails. Write checkpoints and resume for a longer job.

You cannot add a CPU pool with `kinetic pool add`. Every cluster has a
default pool with one `e2-standard-4` node that runs at all times. That
node runs the cluster system pods and `accelerator="cpu"` jobs. The
cluster autoscaler adds CPU nodes when a CPU job needs more.

## Share a cluster with a team

The infrastructure state lives in the state bucket of the project, not
on one laptop. A teammate who has access to the bucket runs
`kinetic init`, sees the cluster on the **Join** path, and gets a
profile for it. No one exports environment variables and no one copies
files.

IAM requirements:

- The first person who runs `kinetic up` in a project needs
  `roles/storage.admin`, because that run creates the state bucket.
- Every other person needs `roles/storage.objectAdmin` on the state
  bucket, and the permissions to submit jobs: read and write access to
  the jobs bucket, and Kubernetes access to the cluster.

See [Security](../security.md) for the trust model.

## Run more than one cluster

Most people need one cluster. Create a second cluster when you have one
of these reasons:

- **Isolation** — GPU jobs and TPU jobs on separate clusters.
- **Location** — jobs in two zones or two regions, for example to find
  Spot capacity.
- **Environments** — separate clusters for development and production.

:::{warning}
Each cluster has its own GKE control plane. The control plane costs about
$0.10 per hour, and the Google Cloud free tier covers one cluster only.
Each cluster also has its own repository and buckets. Do not create a
second cluster without one of the reasons above.
:::

Create a named cluster:

```bash
kinetic up --cluster gpu-cluster --zone us-east1-b --accelerator gpu-a100
```

`kinetic up` saves a profile with the same name as the cluster and makes
that profile active. Switch between clusters with the profile:

```bash
kinetic profile ls
kinetic profile use gpu-cluster
```

For a one-off command against a different cluster, use the `--profile`
flag, or pass both `--cluster` and `--zone`. Kinetic identifies a cluster by
its project, its zone, and its name together:

```bash
kinetic --profile gpu-cluster status
kinetic status --cluster gpu-cluster --zone us-east1-b
```

For a one-off job, pass `cluster=` to the decorator:

```python
@kinetic.run(accelerator="gpu-a100", cluster="gpu-cluster", zone="us-east1-b")
def train_on_gpu(): ...
```

Each cluster has its own set of resources, named after the cluster:

| Resource | Name for `gpu-cluster` |
| -------- | ---------------------- |
| GKE cluster | `gpu-cluster` |
| Artifact Registry repository | `kn-gpu-cluster` |
| Jobs bucket | `{project}-kn-gpu-cluster-jobs` |
| Builds bucket | `{project}-kn-gpu-cluster-builds` |
| Infrastructure stack | `{project}-gpu-cluster` |

## Delete a cluster

```bash
kinetic down
```

`kinetic down` deletes the cluster, the node pools, the Cloud NAT
gateway, and the Artifact Registry repository with its images. It also
deletes the jobs bucket and the builds bucket **with their contents**,
including everything under `KINETIC_OUTPUT_DIR`. Copy the outputs that you want to keep to another
bucket first. `kinetic down` leaves the Google Cloud APIs enabled, and it
does not delete the state bucket or your profiles.

Pass `--cluster NAME --zone ZONE` to delete a cluster other than the
one in the active profile. Kinetic identifies a cluster by its project,
its zone, and its name together. Kinetic does not look up the zone from the
name. The active profile still points at the deleted cluster
afterwards. Run `kinetic profile use` or `kinetic profile rm` to update
it.

By default, `kinetic down` empties the buckets before it deletes them.
Run `kinetic up --no-force-destroy` to store the opposite choice in the
stack. `kinetic down` then fails until you empty the buckets yourself.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1em` Profiles
:link: profiles
:link-type: doc

One profile per cluster; switch with one command.
:::

:::{grid-item-card} {octicon}`cpu;1em` Accelerators
:link: ../accelerators
:link-type: doc

Every accelerator name and topology.
:::

:::{grid-item-card} {octicon}`graph;1em` Cost Optimization
:link: cost_optimization
:link-type: doc

Scale to zero, warm nodes, Spot VMs, and the control plane cost.
:::

:::{grid-item-card} {octicon}`checklist;1em` Capacity Reservations
:link: reservations
:link-type: doc

Guaranteed hardware for a node pool.
:::
::::
