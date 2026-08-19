# PyTorch Training

Kinetic runs a PyTorch function on a cloud GPU node in the same way that
it runs a Keras or a JAX function. You add `torch` to your dependency
file, you decorate the function with a GPU accelerator, and you call the
function. This page shows a first GPU job and a job on several GPUs of
one node. It also shows how to keep outputs and how to run on Spot
capacity. Read [Getting Started](../getting_started.md) before this page.

## Before you start

- You have run `kinetic init`, so an active profile supplies the project,
  the zone, and the cluster. You do not pass those values in code.
- Your cluster has a GPU node pool. List the pools with
  `kinetic pool list`. If no GPU pool exists, add one:

  ```bash
  kinetic pool add --accelerator gpu-l4
  ```

  Use the same accelerator string for the pool and for the job. The pool
  then has the GPU type and the GPU count that the job requests. When a
  cluster has a GPU node pool, Kinetic installs the NVIDIA driver on the
  GPU nodes. You do not install the driver.

## Add `torch` to the dependency file

Put a `requirements.txt` next to your script:

```text
torch
torchvision
```

Kinetic finds the file and installs the packages into the image that it
builds for the job. A `pyproject.toml` with a `[project.dependencies]`
list works too. See [Dependencies](../guides/dependencies.md) for the
rules that select the file.

The image that Kinetic builds for a GPU job starts from a `python` image
that matches your local Python minor version. Kinetic installs JAX with
CUDA support, Keras, `cloudpickle`, `google-cloud-storage`, and the
`keras-kinetic` package into that image. PyTorch is not in that set, so
`torch` must be in your dependency file. The Linux `torch` wheels on
PyPI include the CUDA libraries, and the GPU node supplies the NVIDIA
driver. You do not add CUDA packages to the file.

:::{note}
The pod runs with `KERAS_BACKEND=jax`. If your function uses Keras 3
with the PyTorch backend, set the variable inside the function before
you import Keras:

```python
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
```

If `KERAS_BACKEND=torch` is set in your local shell, you can also pass
`capture_env_vars=["KERAS_BACKEND"]` to the decorator. The exact name is
necessary. A wildcard pattern never captures `KERAS_BACKEND`. See
[Forward Environment Variables](../guides/env_vars.md).
:::

## Run on one GPU

`accelerator="gpu-l4"` requests one NVIDIA L4 GPU. The pod requests one
GPU from Kubernetes, and `torch.cuda.is_available()` returns `True` on
the pod. Keep the `torch` imports inside the function, so that your local
process does not need PyTorch.

```python
import kinetic


@kinetic.run(accelerator="gpu-l4")
def train():
  import torch
  import torch.nn as nn

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Training on: {device}")

  # Simple feedforward network
  model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
  ).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = nn.MSELoss()

  # Dummy data
  x = torch.randn(512, 10, device=device)
  y = torch.randn(512, 1, device=device)

  for epoch in range(20):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
      print(f"epoch {epoch}: loss={loss.item():.4f}")

  return loss.item()


final_loss = train()
```

The call blocks until the job ends. Kinetic streams the `print` output to
your terminal and returns the final loss as a Python float. The first run
needs 5 to 10 minutes before the function starts, because Kinetic builds
the image with `torch` and the autoscaler starts a GPU node. Later runs
with the same dependency file reuse the image and start in less than
1 minute while a node still runs.

For a job that runs longer than a few minutes, call `train.run_async()`
instead. That call returns a `JobHandle` as soon as Kinetic submits the
job. See [Detached Jobs](../guides/async_jobs.md).

## Use several GPUs on one node

Append `xN` to the GPU name to request `N` GPUs on one node. Every GPU
job is single-host: all GPUs of the job are on one VM, and the job runs
in one pod. `gpu-a100x4` requests four A100 40GB GPUs on one
`a2-highgpu-4g` VM. Add a node pool for that string before you run the
job:

```bash
kinetic pool add --accelerator gpu-a100x4
```

On the pod, `torch.cuda.device_count()` returns `4`. Use
`torch.nn.DataParallel` to split each batch across the GPUs:

```python
import kinetic


@kinetic.run(accelerator="gpu-a100x4")
def train_multi_gpu():
  import torch
  import torch.nn as nn

  device = torch.device("cuda")
  print(f"GPUs available: {torch.cuda.device_count()}")

  model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
  )
  model = nn.DataParallel(model).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = nn.MSELoss()

  x = torch.randn(2048, 10, device=device)
  y = torch.randn(2048, 1, device=device)

  for epoch in range(20):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return loss.item()
```

Kinetic starts one Python process in the pod and does not run `torchrun`.
If you use `DistributedDataParallel`, start the worker processes from
inside the function, for example with `torch.multiprocessing.spawn`. Use
`localhost` as the rendezvous address, because all workers are in one
pod. Kinetic does not run a GPU job across more than one node.
Multi-host jobs are available for TPU slices only. See
[Distributed Training](../guides/distributed_training.md).

## Bring data and keep outputs

Two rules from the other examples also apply to PyTorch:

- **Inputs.** Wrap a local file, a local directory, or a `gs://` URI in
  `kinetic.Data(...)`. Kinetic uploads local data one time and gives your
  function a plain path on the pod. See
  [Working with Data](../guides/data.md).
- **Outputs.** Kubernetes deletes the pod and its filesystem when the
  job ends. Write every file that you want to keep under
  `KINETIC_OUTPUT_DIR`, a Cloud Storage prefix that Kinetic sets for
  each job. See [Outputs and Checkpoints](../guides/checkpointing.md).

`KINETIC_OUTPUT_DIR` is a `gs://` URI, and `torch.save` writes to a local
file. Save the file to the local disk of the pod. Then upload the file
with the `google-cloud-storage` client, which the image contains:

```python
import os

import kinetic


@kinetic.run(accelerator="gpu-l4")
def train_and_save():
  import torch
  from google.cloud import storage

  model = ...  # build and train the model

  local_path = "/tmp/model.pt"
  torch.save(model.state_dict(), local_path)

  # KINETIC_OUTPUT_DIR is gs://<jobs bucket>/outputs/<job_id> by default.
  bucket_name, prefix = os.environ["KINETIC_OUTPUT_DIR"][5:].split("/", 1)
  blob = storage.Client().bucket(bucket_name).blob(f"{prefix}/final/model.pt")
  blob.upload_from_filename(local_path)
  return f"gs://{bucket_name}/{prefix}/final/model.pt"
```

The default output directory contains the job ID, so each call gets an
empty directory. If a second call must find the checkpoints of the first
call, pass the same `output_dir=` to the decorator on both calls.

## Select a GPU

The `accelerator` string names the GPU type and, with the `xN` suffix,
the GPU count. Kinetic accepts these GPU names: `l4`, `t4`, `v100`,
`a100`, `a100-80gb`, `h100`, `p4`, and `p100`. The `gpu-` prefix is
optional, so `gpu-l4` and `l4` name the same hardware.
[Accelerators](../accelerators.md) lists the counts and the machine
types for each GPU. `kinetic accelerators` prints the same list in the
shell, and `kinetic accelerators --live` marks each accelerator type
that has a node pool on your cluster.

A job runs only on a node pool with the same GPU type, and each node of
the pool must have at least the requested GPU count. If no pool has the
GPU type, a blocking call stops with the error `No GKE node pool
exists`, and a detached job stays `PENDING`. If a pool has the GPU type
but too few GPUs on each node, the job stays `PENDING`. Add a pool with
`kinetic pool add`, or change the `accelerator` string to a pool that
the cluster has.

## Run on Spot capacity

`spot=True` on the decorator asks for a Spot node. Spot VMs cost less,
but Google Cloud can preempt a Spot VM at any time. Spot use in Kinetic
has two parts, and both parts are necessary:

1. Add a Spot node pool for the accelerator:

   ```bash
   kinetic pool add --accelerator gpu-a100 --spot
   ```

2. Set `spot=True` on the job. The accelerator string
   `"gpu-a100:spot"` has the same effect:

   ```python
   @kinetic.run(accelerator="gpu-a100", spot=True)
   def train(): ...
   ```

The job side adds the Spot node selector and the Spot toleration to the
pod. Without `spot=True`, the pod cannot schedule on the Spot pool, and
the job stays `PENDING`. Without a `--spot` pool, a job with `spot=True`
has no node to run on: a blocking call stops with the error `No GKE node
pool exists`, and a detached job stays `PENDING`.

:::{warning}
If Google Cloud preempts the node, the job ends as `FAILED`, and Kinetic
does not submit the job again. Use Spot only for a job that can restart
from a checkpoint. Write checkpoints under `KINETIC_OUTPUT_DIR`, and pass
a fixed `output_dir=` so that the next submission finds them.
:::

See [Spot VMs](../guides/cost_optimization.md#spot-vms) for the full
rules.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1em` Dependencies
:link: ../guides/dependencies
:link-type: doc

How Kinetic finds your dependency file and installs `torch` into the image.
:::

:::{grid-item-card} {octicon}`cpu;1em` Accelerators
:link: ../accelerators
:link-type: doc

Every GPU name, GPU count, and machine type that Kinetic accepts.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: ../guides/checkpointing
:link-type: doc

`KINETIC_OUTPUT_DIR`, retention, and how to resume a job from a checkpoint.
:::

:::{grid-item-card} {octicon}`zap;1em` Cost Optimization
:link: ../guides/cost_optimization
:link-type: doc

Spot pools, scale to zero, and warm nodes for GPU jobs.
:::
::::
