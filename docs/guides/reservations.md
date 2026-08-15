# Capacity Reservations

Google Cloud does not guarantee on-demand capacity for newer
accelerators, for example TPU v6e and H100. When a node pool scales up
and the zone has no free VM, GKE reports a `FailedScaleUp` event and
the job stays `PENDING`. A capacity reservation guarantees that the
hardware is available when the pool scales up. This page shows how to
get a reservation and how to bind it to a Kinetic node pool. Read this
page if your jobs stay `PENDING` because of capacity, or if you need
hardware at a fixed time.

## Before you start

- You need a cluster and an active profile. See
  [Getting Started](../getting_started.md). The profile supplies the
  project and the zone to `kinetic pool add`.
- A reservation applies to one zone. Create the reservation in the zone
  of your cluster. `kinetic config` prints the zone of the active
  profile.
- A reservation applies to one machine type. The machine type must be
  the type that Kinetic uses for your accelerator. The
  [Accelerators](../accelerators.md) page lists the machine type for each
  accelerator name.

## How a reservation works with Kinetic

`kinetic pool add --reservation NAME` sets a `SPECIFIC_RESERVATION`
affinity on the node pool, with the key
`compute.googleapis.com/reservation-name` and the value `NAME`. The
cluster autoscaler then creates the nodes of that pool from your
reservation only. The pool does not compete for on-demand capacity.

Kinetic does not validate the reservation. Kinetic passes the name to
GKE unchanged. If the name is wrong, the scale-up fails and the job
stays `PENDING`. The same happens if the zone or the machine type of the
reservation does not match the pool.

A reservation has a fixed number of VMs. Make sure that the reservation
covers the number of VMs that the pool uses at one time:

- A GPU job uses one VM. A GPU pool scales up to `--min-nodes` plus 10
  VMs.
- A single-host TPU job uses one VM.
- A multi-host TPU job uses one VM per host. `tpu-v6e-16` uses 4 VMs of
  type `ct6e-standard-4t`. A TPU pool scales up to `--min-nodes` plus
  one slice.

With the default `--min-nodes 0`, the pool creates no VM until a job
runs. No VM consumes the reservation while the pool is idle.

## Step 1: Create a GPU reservation

Create a GPU reservation with the `gcloud` command. This example
reserves one VM with one H100, which is the machine type
`a3-highgpu-1g` for `accelerator="gpu-h100"`:

```bash
gcloud compute reservations create my-h100-reservation \
  --machine-type=a3-highgpu-1g \
  --vm-count=1 \
  --zone=us-central1-a \
  --project=your-project-id
```

`gcloud` does not read the Kinetic profile. The `gcloud` commands on
this page therefore name the project and the zone. Use the zone of your
cluster.

Two options are useful:

- `--vm-count=N` reserves N VMs. Use the number of VMs from the section
  above.
- `--require-specific-reservation` makes sure that only a VM that names
  the reservation can consume it. Without this option, any VM of the
  same machine type in the project can consume the reservation.

For a machine type in the `n1` family, for example `n1-standard-4` for
one T4, add the accelerator to the reservation:
`--accelerator=type=nvidia-tesla-t4,count=1`.

See the [Compute Engine reservations
documentation](https://cloud.google.com/compute/docs/instances/reservations-overview)
for the full list of machine types and options.

## Step 2: Request a TPU reservation

Google Cloud provides TPU capacity as a *future reservation*, not
through `gcloud compute reservations create`. Two paths exist:

- For a term of up to 90 days, request a future reservation in calendar
  mode yourself, with the Google Cloud CLI or the console. Google Cloud
  supports TPU v5p, TPU v6e, and TPU7x in calendar mode.
- For a longer term, or for a TPU type that calendar mode does not
  support, contact Cloud Sales or your account team.

See [About Cloud TPU
reservations](https://docs.cloud.google.com/tpu/docs/about-tpu-reservations)
for the current list of TPU types and terms. In the request, give the
zone, the TPU version, and the number of chips. For example,
`tpu-v6e-16` needs 16 TPU v6e chips, which is 4 VMs of type
`ct6e-standard-4t`, in one zone.

When the reservation period starts, Google Cloud creates the reservation
in your project. Find the name and the zone with:

```bash
gcloud compute reservations list --project=your-project-id
```

## Step 3: Bind the reservation to a node pool

Pass the reservation name to `kinetic pool add`. The active profile
supplies the project and the zone:

```bash
kinetic pool add \
  --accelerator gpu-h100 \
  --reservation my-h100-reservation
```

For a TPU reservation, use the matching TPU accelerator name:

```bash
kinetic pool add \
  --accelerator tpu-v6e-16 \
  --reservation my-v6e-reservation
```

The `KINETIC_RESERVATION` environment variable sets the same value for
one command. See [Configuration](../configuration.md).

:::{note}
You cannot combine `--reservation` with `--spot`. A Spot VM cannot
consume a reservation, and `kinetic pool add` rejects the two flags
together. Create a separate pool for Spot jobs.
:::

Every job that lands on the pool uses the reservation. You do not
change the decorator. A job with `@kinetic.run(accelerator="gpu-h100")`
runs on any pool of the cluster with that accelerator. If the cluster has
a reserved H100 pool and a second H100 pool without a reservation, a job
can land on either pool. Keep one pool per accelerator, unless the pools
differ in Spot or in reservation and you accept that a job can land on
either pool. See [Clusters and Node Pools](clusters.md).

:::{warning}
Kinetic does not save the reservation name, or the Spot setting, of a
pool in its infrastructure state.
A later `kinetic pool add`, `kinetic pool remove`, or `kinetic up`
re-applies every existing pool from that state, without the
reservation. The reserved pool then loses its reservation affinity.
After you run one of these commands, remove the reserved pool and add
it again with `--reservation`.
:::

## Check that a job uses the reservation

Run a job with the accelerator of the pool. If the job stays `PENDING`
for more than 10 minutes, do these checks:

1. Describe the reservation:

   ```bash
   gcloud compute reservations describe my-h100-reservation \
     --zone=us-central1-a \
     --project=your-project-id
   ```

   Make sure that the zone and the machine type match the pool, and
   that the reservation has a free VM.
2. Run `kinetic pool list`. Make sure that a pool for the accelerator
   exists.
3. See [Troubleshooting](../troubleshooting.md) for the quota checks.

## Clean up

Google Cloud bills a reservation for the reserved machine type while
the reservation exists, also when no VM uses it. Delete the reservation
when you no longer need the hardware:

```bash
gcloud compute reservations delete my-h100-reservation \
  --zone=us-central1-a \
  --project=your-project-id
```

Also remove the node pool that names the reservation. After you delete
the reservation, that pool cannot create nodes:

```bash
kinetic pool list
kinetic pool remove gpu-h100-1a2b
```

:::{note}
The reservation charge starts when Google Cloud creates the reservation.
The charge stops when you delete the reservation, or when a future
reservation reaches its end time. A pool with `--min-nodes 0` does not
stop the charge, because Google Cloud holds the reserved VMs for you.
:::

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1em` Cost Optimization
:link: cost_optimization
:link-type: doc

When to select a reservation, Spot VMs, or on-demand capacity.
:::

:::{grid-item-card} {octicon}`server;1em` Clusters and Node Pools
:link: clusters
:link-type: doc

Add, list, and remove node pools, and run more than one cluster.
:::

:::{grid-item-card} {octicon}`cpu;1em` Accelerators
:link: ../accelerators
:link-type: doc

The machine type and the host count behind each accelerator name.
:::
::::
