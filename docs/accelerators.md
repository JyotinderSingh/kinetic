# Accelerators

The `accelerator` argument of `@kinetic.run()` and the `--accelerator`
flag of `kinetic pool add` take the same strings. This page lists every
name that Kinetic accepts, and the topology behind each TPU name.

:::{important}
A job runs only on a node pool with the same accelerator type and, for
TPUs, the same topology. Add one node pool for each accelerator that you
use. See [Clusters and Node Pools](guides/clusters.md).
:::

Run `kinetic accelerators` to print the names, the chip counts, and the
topologies (TPUs) or the counts and machine types (GPUs) from your
installed version. This page adds the machine type and the host count for
each TPU slice. Add `--live` to mark the accelerators that have a node
pool on your cluster.

## Name formats

| Format | Examples | Meaning |
| ------ | -------- | ------- |
| CPU | `cpu` | A CPU-only node. |
| GPU name | `gpu-l4`, `l4`, `gpu:l4` | One GPU of that type. |
| GPU name with count | `gpu-a100x4`, `a100x4` | That many GPUs on one node. |
| TPU name with chip count | `tpu-v5litepod-8`, `v5litepod-8`, `tpu:v5litepod-8` | A slice with that many chips. |
| TPU name with topology | `tpu-v5litepod-2x4`, `tpu-v4-2x2x2` | The same slice, named by topology. |
| TPU name only | `tpu-v5litepod`, `tpu-v6e` | The default chip count of that type: `v5litepod-4`, `v6e-8`, `v5p-8`, `v4-4`, `v3-4`. |
| Count only | `gpu:4`, `tpu:8` | The most capable type that supports that count. GPUs: `h100`, then `a100-80gb`, `a100`, `l4`, `v100`, `t4`, `p100`, `p4`. TPUs: `v6e`, then `v5p`, `v5litepod`, `v4`, `v3`. |
| Spot suffix | `gpu-l4:spot` | The same as `spot=True`. |

The `gpu-` and `tpu-` prefixes are optional. `v5e` is an alias for
`v5litepod`. Names are not case-sensitive.

## TPUs

The **Hosts** column decides the backend. A slice with more than one
host is multi-host: Kinetic runs it on the `pathways` backend, one pod per
host, and startup takes longer. See
[Distributed Training](guides/distributed_training.md).

| Type | Name | Topology | Machine type | Hosts |
| ---- | ---- | -------- | ------------ | ----- |
| TPU v6e | `v6e-8` | 2x4 | `ct6e-standard-4t` | 2 |
| | `v6e-16` | 4x4 | `ct6e-standard-4t` | 4 |
| TPU v5p | `v5p-8` | 2x2x2 | `ct5p-hightpu-4t` | 2 |
| | `v5p-16` | 2x2x4 | `ct5p-hightpu-4t` | 4 |
| | `v5p-32` | 2x4x4 | `ct5p-hightpu-4t` | 8 |
| TPU v5e (`v5litepod`) | `v5litepod-1` | 1x1 | `ct5lp-hightpu-1t` | 1 |
| | `v5litepod-4` | 2x2 | `ct5lp-hightpu-4t` | 1 |
| | `v5litepod-8` | 2x4 | `ct5lp-hightpu-8t` | 1 |
| | `v5litepod-16` | 4x4 | `ct5lp-hightpu-4t` | 4 |
| | `v5litepod-32` | 4x8 | `ct5lp-hightpu-4t` | 8 |
| | `v5litepod-64` | 8x8 | `ct5lp-hightpu-4t` | 16 |
| | `v5litepod-128` | 8x16 | `ct5lp-hightpu-4t` | 32 |
| | `v5litepod-256` | 16x16 | `ct5lp-hightpu-4t` | 64 |
| TPU v4 | `v4-4` | 2x2x1 | `ct4p-hightpu-4t` | 1 |
| | `v4-8` | 2x2x2 | `ct4p-hightpu-4t` | 2 |
| | `v4-16` | 2x2x4 | `ct4p-hightpu-4t` | 4 |
| | `v4-32` | 2x4x4 | `ct4p-hightpu-4t` | 8 |
| | `v4-64` | 4x4x4 | `ct4p-hightpu-4t` | 16 |
| | `v4-128` | 4x4x8 | `ct4p-hightpu-4t` | 32 |
| | `v4-256` | 4x8x8 | `ct4p-hightpu-4t` | 64 |
| | `v4-512` | 8x8x8 | `ct4p-hightpu-4t` | 128 |
| | `v4-1024` | 8x8x16 | `ct4p-hightpu-4t` | 256 |
| | `v4-2048` | 8x16x16 | `ct4p-hightpu-4t` | 512 |
| | `v4-4096` | 16x16x16 | `ct4p-hightpu-4t` | 1024 |
| TPU v3 | `v3-4` | 2x2 | `ct3-hightpu-4t` | 1 |
| | `v3-16` | 4x4 | `ct3p-hightpu-4t` | 4 |
| | `v3-32` | 4x8 | `ct3p-hightpu-4t` | 8 |
| | `v3-64` | 8x8 | `ct3p-hightpu-4t` | 16 |
| | `v3-128` | 8x16 | `ct3p-hightpu-4t` | 32 |
| | `v3-256` | 16x16 | `ct3p-hightpu-4t` | 64 |
| | `v3-512` | 16x32 | `ct3p-hightpu-4t` | 128 |
| | `v3-1024` | 32x32 | `ct3p-hightpu-4t` | 256 |
| | `v3-2048` | 32x64 | `ct3p-hightpu-4t` | 512 |

For a first TPU job, use a single-host slice: `v5litepod-1`,
`v5litepod-4`, or `v5litepod-8`.

## GPUs

| Type | Names | GPU counts | Machine types |
| ---- | ----- | ---------- | ------------- |
| NVIDIA H100 80GB | `h100`, `nvidia-h100-80gb` | 1, 2, 4, 8 | `a3-highgpu-{1,2,4,8}g` |
| NVIDIA A100 80GB | `a100-80gb`, `nvidia-a100-80gb` | 1, 2, 4, 8, 16 | `a2-ultragpu-{1,2,4,8,16}g` |
| NVIDIA A100 40GB | `a100`, `nvidia-tesla-a100` | 1, 2, 4, 8, 16 | `a2-highgpu-{1,2,4,8}g`, `a2-megagpu-16g` |
| NVIDIA L4 | `l4`, `nvidia-l4` | 1, 2, 4, 8 | `g2-standard-{4,24,48,96}` |
| NVIDIA V100 | `v100`, `nvidia-tesla-v100` | 1, 2, 4, 8 | `n1-standard-{8,16,32,64}` |
| NVIDIA T4 | `t4`, `nvidia-tesla-t4` | 1, 2, 4 | `n1-standard-{4,8,16}` |
| NVIDIA P100 | `p100`, `nvidia-tesla-p100` | 1, 2, 4 | `n1-standard-{4,8,16}` |
| NVIDIA P4 | `p4`, `nvidia-tesla-p4` | 1, 2, 4 | `n1-standard-{4,8,16}` |

Append `xN` for more than one GPU on one node: `a100x4`, `l4x2`. Every
GPU job is single-host.

## CPU

`accelerator="cpu"` runs the job on a CPU node without an accelerator.
Every cluster can run CPU jobs. Use `cpu` to test a script before you
request hardware.

## Capacity

Newer accelerators, such as TPU v6e and H100, can have no on-demand
capacity in a zone. The job then stays `PENDING`. A capacity reservation
guarantees the hardware for a node pool:

```bash
kinetic pool add --accelerator tpu-v6e-16 --reservation my-v6e-reservation
```

See [Capacity Reservations](guides/reservations.md).

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`server;1em` Clusters and Node Pools
:link: guides/clusters
:link-type: doc

Add a node pool for each accelerator that you use.
:::

:::{grid-item-card} {octicon}`cpu;1em` Distributed Training
:link: guides/distributed_training
:link-type: doc

What changes when a slice spans more than one host.
:::

:::{grid-item-card} {octicon}`graph;1em` Cost Optimization
:link: guides/cost_optimization
:link-type: doc

Spot VMs, reservations, and on-demand capacity.
:::
::::
