# Native JAX Training

This page shows how to run a training loop that you write directly in
JAX on a cloud TPU or GPU. Read this page if you use `jax.grad`,
`jax.pmap`, or `jax.sharding` without Keras. Kinetic runs a JAX function
the same way that it runs a Keras function: you decorate the function
with `@kinetic.run()` and call it. The sections below cover the
JAX-specific details: the JAX runtime in the image, single-host
parallelism, multi-host slices, data, and outputs.

## Before you start

- Complete [Getting Started](../getting_started.md). You need an active
  profile and a cluster.
- Make sure that the cluster has a node pool for the accelerator that
  you use. `kinetic pool list` shows the pools of the cluster.
  `kinetic pool add --accelerator tpu-v5litepod-8` adds a pool for the
  first example. See [Clusters and Node Pools](../guides/clusters.md).
- Do not add `jax`, `jaxlib`, or `libtpu` to your dependency file.
  Kinetic installs them for you. See
  [The JAX runtime in the image](#the-jax-runtime-in-the-image).

## A first run

Save this script and run it with `python`:

```python
import kinetic


@kinetic.run(accelerator="tpu-v5litepod-8")
def jax_computation():
  import jax
  import jax.numpy as jnp

  print(f"Devices: {jax.devices()}")

  x = jnp.ones((1000, 1000))
  result = jnp.dot(x, x)
  return float(result[0, 0])


print(jax_computation())  # 1000.0
```

Kinetic sends the function to the cluster, streams the log lines to your
terminal, and returns the value to the script. The first run takes 5 to 10
minutes, because Kinetic builds a container image with your
dependencies. Later runs with the same dependency file start in less
than 1 minute while a node still runs.

A training loop with `jax.grad` runs without a change to the loop:

```python
@kinetic.run(accelerator="tpu-v5litepod-4")
def train():
  import jax
  import jax.numpy as jnp

  def loss_fn(params, x, y):
    pred = x @ params["w"] + params["b"]
    return jnp.mean((pred - y) ** 2)

  grad_fn = jax.grad(loss_fn)

  key = jax.random.PRNGKey(0)
  params = {"w": jax.random.normal(key, (10, 1)), "b": jnp.zeros(1)}
  x = jax.random.normal(key, (512, 10))
  y = x @ jnp.ones((10, 1)) + 0.1 * jax.random.normal(key, (512, 1))

  lr = 0.01
  for step in range(200):
    grads = grad_fn(params, x, y)
    params = {k: params[k] - lr * grads[k] for k in params}
    if step % 50 == 0:
      print(f"step {step}: loss={loss_fn(params, x, y):.4f}")

  return float(loss_fn(params, x, y))
```

Put `import jax` and the imports of other heavy libraries **inside** the
decorated function. The import then runs on the pod, which has the JAX
build for the accelerator. Your local machine does not need JAX.

## The JAX runtime in the image

JAX needs a `jaxlib` build and an accelerator runtime that match the
hardware. The image that Kinetic builds contains JAX for your
accelerator:

| Accelerator | JAX package in the image |
| ----------- | ------------------------ |
| TPU (`tpu-...`) | `jax[tpu]`, which includes `libtpu` |
| GPU (`gpu-...`) | `jax[cuda12]`, which includes the CUDA libraries |
| CPU (`cpu`) | `jax` |

Do not pin `jax`, `jaxlib`, or `libtpu` in your dependency file.
Kinetic removes the entries for `jax`, `jaxlib`, `libtpu`, and
`libtpu-nightly` from the dependency list before the install. A pin in
your file therefore does not replace the installation in the image.
Kinetic logs a warning for each removed entry. To keep a line in
`requirements.txt`, append `# kn:keep` to the line. See
[JAX and accelerator runtimes](../guides/dependencies.md#jax-and-accelerator-runtimes)
for the filter rules and the override.

Inside the function, `jax.devices()` returns the devices of the pod:

- `tpu-v5litepod-8`: 8 TPU devices on one host.
- `tpu-v5litepod-4`: 4 TPU devices on one host.
- `gpu-l4`: one GPU device.
- `cpu`: one CPU device.

## Single-host parallelism

A single-host slice, for example `tpu-v5litepod-8`, holds all of its
chips on one VM. `jax.pmap` or `jax.sharding` spreads the computation
across those chips. Kinetic needs no extra setting for this case.

```python
@kinetic.run(accelerator="tpu-v5litepod-8")
def parallel_computation():
  import jax
  import jax.numpy as jnp

  n_devices = jax.local_device_count()
  print(f"Running on {n_devices} devices")

  @jax.pmap
  def parallel_matmul(x):
    return jnp.dot(x, x.T)

  data = jnp.ones((n_devices, 256, 256))
  result = parallel_matmul(data)
  return float(result[0, 0, 0])
```

On `tpu-v5litepod-8`, `jax.local_device_count()` is 8, and each `pmap`
replica runs on one chip.

## Multi-host slices

Some slices span more than one host. `tpu-v5litepod-16` and `tpu-v6e-16`
each consist of four 4-chip VMs. Kinetic reads the host count from the
accelerator name and selects the `pathways` backend for you. You do not
set `backend=`. Kinetic runs one pod per host and sets the environment
variables for multi-controller JAX on every pod. JAX then starts one
process per host. `jax.process_count()` equals the host count,
`jax.local_device_count()` is the chip count of one host, and
`jax.device_count()` is the total. Collectives across hosts work, for
example `jax.lax.psum` inside `pmap`.

```python
@kinetic.run(accelerator="tpu-v6e-16")
def train_distributed():
  import jax

  print(f"Host {jax.process_index()} of {jax.process_count()}")
  print(f"Devices on this host: {jax.local_device_count()}")
  print(f"Total devices: {jax.device_count()}")
  # pmap and sharding work across hosts here.
  ...
```

On `tpu-v6e-16`, `jax.process_count()` is 4, `jax.local_device_count()`
is 4, and `jax.device_count()` is 16.

A multi-host job has two requirements:

- The cluster needs a node pool for the multi-host accelerator, for
  example `kinetic pool add --accelerator tpu-v6e-16`.
- The cluster needs the LeaderWorkerSet controller. `kinetic up`
  installs it.

:::{note}
The chip count alone does not tell you the host count. `tpu-v5litepod-8`
is one 8-chip VM, but `tpu-v6e-8` is two 4-chip VMs and therefore a
multi-host job. The **Hosts** column on the
[Accelerators](../accelerators.md#tpus) page decides.
:::

See [Distributed Training](../guides/distributed_training.md) for the
log that you see, the return value, and the failure behavior of a
multi-host job.

## Data

To pass a dataset into a remote JAX function, construct a
`kinetic.Data(...)` object **at the call site** in your local script.
Pass the object as an argument. Kinetic uploads or mounts the source and
gives the remote function a plain filesystem path. The decorated
function only sees a `str` path:

```python
import kinetic
from kinetic import Data


@kinetic.run(accelerator="tpu-v5litepod-8")
def train(data_dir):
  # `data_dir` is a local filesystem path on the remote pod.
  import os

  files = os.listdir(data_dir)
  ...


# Local directory:
train(Data("./my_dataset/"))

# Existing Cloud Storage prefix (the trailing slash marks a directory):
train(Data("gs://my-bucket/dataset/"))

# Large Cloud Storage dataset, read on demand through FUSE:
train(Data("gs://my-bucket/large/", fuse=True))
```

`Data` accepts a local path, a `gs://` URI, or an `hf://` URI for a
Hugging Face dataset. See [Working with Data](../guides/data.md) for the
choice between a downloaded copy, a FUSE mount, and direct `gs://`
access.

## Outputs and checkpoints

Kubernetes deletes the pod filesystem, including `/tmp`, when the pod
ends. Write every file that you want to keep under `KINETIC_OUTPUT_DIR`.
Kinetic sets that environment variable in the pod to a Cloud Storage
prefix that stays after the pod ends. Orbax writes to a `gs://` path
directly, so pass that prefix to the `CheckpointManager`:

```python
@kinetic.run(accelerator="tpu-v5litepod-8")
def train():
  import os

  import orbax.checkpoint as ocp

  output_dir = os.environ["KINETIC_OUTPUT_DIR"]
  mngr = ocp.CheckpointManager(
    f"{output_dir}/checkpoints", ocp.StandardCheckpointer()
  )
  ...
```

Add `orbax-checkpoint` to your dependency file for this example.

See [Outputs and Checkpoints](../guides/checkpointing.md) for a full
Orbax example, the default location, and how to resume a job.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`server;1em` Distributed Training
:link: ../guides/distributed_training
:link-type: doc

How a multi-host JAX job runs, what the log shows, and how it fails.
:::

:::{grid-item-card} {octicon}`package;1em` Dependencies
:link: ../guides/dependencies
:link-type: doc

The dependency file, the JAX filter, and the `# kn:keep` override.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: ../guides/checkpointing
:link-type: doc

`KINETIC_OUTPUT_DIR`, retention, and Orbax checkpoints that resume.
:::

:::{grid-item-card} {octicon}`database;1em` Working with Data
:link: ../guides/data
:link-type: doc

Ship local files and read Cloud Storage data from your function.
:::
::::
