# Training Keras Models

This page is for a reader who has a Keras training script and wants to
run that script on a cloud TPU or GPU. Kinetic runs your
`model.compile()` and `model.fit()` code on a remote accelerator after
one decorator change. You do not restructure the training loop. This
page shows the first run, explains what the function sees on the pod,
and covers the Keras backend, multi-host slices, data, and outputs.

## Before you start

- Run `kinetic init` one time. The active profile supplies the project,
  the zone, and the cluster to every job. See
  [Getting Started](../getting_started.md).
- Make sure that the cluster has a node pool for the accelerator in the
  example. `kinetic pool list` shows the pools. Add one with
  `kinetic pool add --accelerator tpu-v5litepod-4`, or change
  `accelerator=` to an accelerator that the cluster has.

## A first run

```python
import kinetic


@kinetic.run(accelerator="tpu-v5litepod-4")
def train_model():
  import keras
  import numpy as np

  model = keras.Sequential(
    [
      keras.layers.Dense(64, activation="relu", input_shape=(10,)),
      keras.layers.Dense(1),
    ]
  )
  model.compile(optimizer="adam", loss="mse")

  x_train = np.random.randn(1000, 10)
  y_train = np.random.randn(1000, 1)

  history = model.fit(x_train, y_train, epochs=5, verbose=0)
  return history.history["loss"][-1]


final_loss = train_model()
print(f"Final loss: {final_loss}")
```

Three points about this script:

- Put the imports for `keras` and `numpy` inside the function. Your
  machine then does not need those packages. The pod imports them from
  the image, and the image contains JAX with the runtime for the
  accelerator.
- The return value goes back to your local process. Keep the value
  small: a final metric, a dict of numbers, or a path under
  `KINETIC_OUTPUT_DIR`. Do not return the model object.
- `accelerator="tpu-v5litepod-4"` selects a 4-chip TPU v5e slice on one
  host. Use `accelerator="cpu"` while you develop the code. Change the
  accelerator when the code works. See
  [Accelerators](../accelerators.md).

The first run takes 5 to 10 minutes, because Kinetic builds the
container image with your dependencies. Later runs with the same
dependencies start in less than 1 minute while a node still runs. See
[How Kinetic Works](../concepts.md).

For an end-to-end example with a real dataset, see
[`fashion_mnist.py`](fashion_mnist.md).

## What the function sees

Your decorated function runs in a new Python process, inside a
container, on a node of the cluster. Kinetic serializes the function
with `cloudpickle`, together with the objects that the function
references. Two consequences follow:

- **The payload carries every referenced object.** A value from the
  enclosing scope or a module-level global goes into the payload. A
  small value, such as a config dict, is not a problem. A large object,
  such as a dataset that you load at module level, makes the payload
  large on every submit. Kinetic logs a warning when the payload is
  larger than 50 MB. Load large data inside the function, or pass the
  data as [`kinetic.Data`](../guides/data.md). See
  [What Ships to the Pod](../guides/packaging.md).
- **The image supplies the packages.** The image contains Keras, JAX,
  and the packages from your `requirements.txt` or `pyproject.toml`. A
  `pip install` in your local shell does not carry over. See
  [Dependencies](../guides/dependencies.md).

### The Keras backend

The image that Kinetic builds sets `KERAS_BACKEND=jax`. Keras therefore
uses the JAX backend on the pod, regardless of the backend on your
machine. JAX is also the only accelerator runtime in the image:
`jax[tpu]` for a TPU and `jax[cuda12]` for a GPU.

If your script needs another backend, do three things:

1. Add the framework, for example `torch`, to your dependency file.
2. Set `KERAS_BACKEND` in your shell, for example
   `export KERAS_BACKEND=torch`.
3. Forward the variable with `capture_env_vars`.

```python
@kinetic.run(accelerator="gpu-l4", capture_env_vars=["KERAS_BACKEND"])
def train(): ...
```

Name the variable exactly. A wildcard such as `"KERAS_*"` never
captures `KERAS_BACKEND`, because that variable is on the wildcard
blocklist. The pod applies the captured value before it calls your
function, so the `import keras` inside the function sees the new
backend. See
[Forward Environment Variables](../guides/env_vars.md).

## Scale to more than one host

A single-host slice such as `tpu-v5litepod-8` has up to 8 chips on one
VM. If the model or the batch does not fit on one host, select a
multi-host slice, for example `tpu-v5litepod-16` or `tpu-v6e-16`. Each
of those slices has four 4-chip VMs. Kinetic reads the host count from
the accelerator name and selects the Pathways backend for you. You do
not set `backend="pathways"`.

The Keras distribution API sees every chip on every host. Set a
`DataParallel` distribution before you build the model:

```python
@kinetic.run(accelerator="tpu-v5litepod-16")
def train_distributed():
  import keras

  devices = keras.distribution.list_devices()
  mesh = keras.distribution.DeviceMesh(
    shape=(len(devices),), axis_names=["batch"], devices=devices
  )
  keras.distribution.set_distribution(
    keras.distribution.DataParallel(device_mesh=mesh)
  )

  model = keras.Sequential([...])
  model.compile(...)
  model.fit(...)
```

The cluster needs a node pool for the multi-host accelerator. See
[Distributed Training](../guides/distributed_training.md) for the node
pool, the log that you see, and the return value of a multi-host job.
See [Fine-tuning LLMs](llm_finetuning.md) for a Gemma example.

## Data

Random NumPy arrays inside the function are sufficient for a test. Real
data must reach the pod. Construct a `kinetic.Data(...)` object **at the
call site** in your local script. Pass the object as an argument.
Kinetic uploads a local path one time and downloads the data to the
pod, or mounts a Cloud Storage location. Your function receives a plain
filesystem path (`str`):

```python
import kinetic
from kinetic import Data


@kinetic.run(accelerator="tpu-v5litepod-8")
def train(data_dir):
  # `data_dir` is a local filesystem path on the pod.
  import keras

  ...


# A local directory:
train(Data("./my_dataset/"))

# A directory in Cloud Storage (the trailing slash marks a directory):
train(Data("gs://my-bucket/dataset/"))

# A large Cloud Storage dataset, read on demand through a FUSE mount:
train(Data("gs://my-bucket/large/", fuse=True))

# A Hugging Face dataset:
train(Data("hf://imdb?split=train"))
```

`Data` accepts a local path, a `gs://` URI, or an `hf://` Hugging Face
dataset URI. An `hf://` URI needs the `datasets` package in your
dependency file. See [Working with Data](../guides/data.md) for the
choice between a download, a FUSE mount, and direct `gs://` access, and
for the limits.

## Save the model

Kubernetes deletes the pod filesystem, including `/tmp`, when the pod
ends. Write the files that you want to keep, for example the model
weights and the checkpoints, under `KINETIC_OUTPUT_DIR`. Kinetic sets
that variable on the pod to a per-job Cloud Storage location. Return the
path, not the model:

```python
import os

import kinetic


@kinetic.run(accelerator="tpu-v5litepod-4")
def train():
  import keras

  output_dir = os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/local_run")
  # ... build, compile, and fit the model ...
  # ... write the weights and the metrics under output_dir ...
  return output_dir
```

See [Outputs and Checkpoints](../guides/checkpointing.md) for the output
directory, retention, and a resumable Keras run with Orbax
([`example_keras_checkpoint.py`](example_keras_checkpoint.md)).

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1em` fashion_mnist.py
:link: fashion_mnist
:link-type: doc

A complete example with a real dataset on a TPU.
:::

:::{grid-item-card} {octicon}`database;1em` Working with Data
:link: ../guides/data
:link-type: doc

Ship local files and read Cloud Storage data from your function.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: ../guides/checkpointing
:link-type: doc

`KINETIC_OUTPUT_DIR`, retention, and resumable training.
:::

:::{grid-item-card} {octicon}`cpu;1em` Fine-tuning LLMs
:link: llm_finetuning
:link-type: doc

Keras Hub, Kaggle credentials, and LoRA on Gemma.
:::
::::
