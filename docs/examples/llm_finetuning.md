# Fine-tuning LLMs

This page shows the Kinetic patterns for the fine-tuning of a large
language model from [Keras Hub](https://keras.io/keras_hub/) on a TPU.
It covers four things: the dependency file, the Kaggle credentials,
LoRA on one TPU host, and the move to a multi-host slice. Read this page
before your first fine-tuning job. For a complete tutorial with a saved
model and an inference step, see
[Fine-tuning Gemma 4 on TPU](gemma4_finetuning.md).

## Before you start

- A Kinetic cluster and an active profile. `kinetic init` creates both.
  See [Getting Started](../getting_started.md).
- A TPU node pool that matches the accelerator in your decorator.
  `kinetic pool list` shows the pools of the cluster.
  `kinetic pool add --accelerator tpu-v5litepod-8` adds one. See
  [Clusters and Node Pools](../guides/clusters.md).
- A [Kaggle](https://www.kaggle.com/) account. Accept the terms of the
  model on its Kaggle page before the first download.
- `KAGGLE_USERNAME` and `KAGGLE_KEY` set in your local shell.

## Add `keras-hub` to your dependency file

The image that Kinetic builds contains JAX, Keras, and Kinetic. The image
does not contain `keras-hub`. Put a `requirements.txt` next to your
script. List the packages that your function imports:

```text
# requirements.txt
keras-hub
```

Kinetic finds the file, installs the packages into the image, and caches
the image. Later runs with the same file reuse the image. If your
function imports `keras_hub` and the file does not list `keras-hub`, the
job fails on the pod with `ModuleNotFoundError`. Pin the versions to get
the same packages each time Kinetic builds the image. The Gemma 4
tutorial pins `keras==3.15.0` and `keras-hub==0.27.1`. See
[Dependencies](../guides/dependencies.md).

Two more points about the image:

- Do not list `jax`, `jaxlib`, or `libtpu`. Kinetic filters those lines
  and installs the JAX version that matches the accelerator.
- The pod already has `KERAS_BACKEND=jax`. You do not have to set the
  Keras backend in your function.

## Forward the Kaggle credentials

Keras Hub downloads model presets from Kaggle. The download reads
`KAGGLE_USERNAME` and `KAGGLE_KEY` from the environment. The pod does not
see the environment variables of your shell. List the names in
`capture_env_vars`. Kinetic copies the values into the pod before your
function runs.

```python
import kinetic


@kinetic.run(
  accelerator="tpu-v5litepod-8",
  capture_env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY"],
)
def train_gemma():
  import keras_hub

  # The Kaggle credentials are set in the pod environment.
  gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")
  # ...
```

A name that ends with `*` is a prefix pattern. `capture_env_vars=["KAGGLE_*"]`
forwards every variable with the prefix `KAGGLE_`. See
[Forward Environment Variables](../guides/env_vars.md) for the wildcard
rules and the names that a wildcard never matches.

:::{warning}
Kinetic stores the captured values in plain text in the job payload in
the jobs bucket. Every job pod in the cluster can read that bucket.
Forward only the variables that the job needs. Use tokens with a short
life if you can. Kinetic logs a warning if a captured name contains
`KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or `CREDENTIAL` in any letter
case. The warning is information only. If you intend to forward the
credential, take no action. See [Security](../security.md).
:::

## Fine-tune with LoRA

A full fine-tune updates every weight of the model, and the optimizer
keeps a state for every weight. Those states need a lot of memory, so a
full fine-tune needs a large slice. LoRA (Low-Rank Adaptation) freezes
the model weights and trains small additional matrices of a low rank.
LoRA reduces the number of trainable parameters, so the fine-tune fits
on a smaller slice.

Call `enable_lora(rank=...)` on the backbone before `fit`:

```python
@kinetic.run(
  accelerator="tpu-v5litepod-8",
  capture_env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY"],
)
def train_lora():
  import keras_hub

  gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")

  # Freeze the backbone and add rank-4 LoRA matrices.
  gemma_lm.backbone.enable_lora(rank=4)

  # A short sequence length keeps the activation memory small.
  gemma_lm.preprocessor.sequence_length = 128

  train_data = [
    "Question: What is the capital of India? Answer: New Delhi.",
    "Question: What is the capital of South Africa? Answer: Pretoria.",
  ]
  gemma_lm.fit(train_data, epochs=3)
  return "Training complete"
```

The `rank` value sets the size of the LoRA matrices. A larger rank
trains more parameters and needs more memory. The Gemma 4 tutorial uses
`rank=4` for a 26B-parameter model on `tpu-v5litepod-8`.

## Save the adapted weights

Kubernetes deletes the pod filesystem when the pod ends. Write the LoRA
weights and every other file that you want to keep under
`KINETIC_OUTPUT_DIR`. Kinetic sets that variable in every pod to a Cloud
Storage location, by default `gs://{jobs bucket}/outputs/{job_id}`.
Kinetic does not delete those files as part of the job cleanup. But a
lifecycle rule on the jobs bucket deletes objects that are older than
30 days. See
[Outputs and Checkpoints](../guides/checkpointing.md). The
[Gemma 4 tutorial](gemma4_finetuning.md) shows a complete save step and
a second job that loads the weights for inference.

## Distributed fine-tuning

If the model or the batch does not fit on one host, select a multi-host
slice, for example `tpu-v6e-16` or `tpu-v5litepod-16`. Each of these
slices has 16 chips on four hosts. Kinetic reads the host count from the
accelerator name and selects the `pathways` backend for you. You do not
have to set `backend`. Kinetic runs one pod per host, and JAX sees the
devices of every host.

The Keras distribution API needs a device mesh over all devices. Build
the mesh inside the function. Then load the model:

```python
@kinetic.run(
  accelerator="tpu-v6e-16",
  capture_env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY"],
)
def train_distributed():
  import keras
  import keras_hub

  # Every device on every host.
  devices = keras.distribution.list_devices()
  device_mesh = keras.distribution.DeviceMesh(
    shape=(len(devices),),
    axis_names=["batch"],
    devices=devices,
  )
  keras.distribution.set_distribution(
    keras.distribution.DataParallel(device_mesh=device_mesh)
  )

  gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
  gemma_lm.backbone.enable_lora(rank=4)
  gemma_lm.preprocessor.sequence_length = 128

  train_data = [...]  # one string per example
  gemma_lm.fit(train_data, batch_size=len(devices), epochs=3)
  return "Training complete"
```

Three things change when a job runs on more than one host:

- The cluster needs a node pool for the slice, for example
  `kinetic pool add --accelerator tpu-v6e-16`.
- Your terminal shows the log of the leader host only.
- Every host uploads a return value, and Kinetic returns the value of
  the last host that wrote. Return the same value from every host.

:::{warning}
If your model and your batch fit on one host, stay on one host. A
multi-host job starts more slowly, needs the LeaderWorkerSet controller,
and fails as a whole if one host fails. See
[Distributed Training](../guides/distributed_training.md) before you
move to a multi-host slice.
:::

## Complete examples

- [Fine-tuning Gemma 4 on TPU](gemma4_finetuning.md) — a full tutorial:
  LoRA on `tpu-v5litepod-8`, weights saved under `KINETIC_OUTPUT_DIR`,
  and inference in a second job.
- [Single-TPU Gemma 3 fine-tune](gemma3_sft_demo.md) — the shortest
  script: `gemma3_1b` on `tpu-v5litepod-1` with the Kaggle credentials
  forwarded.
- [Distributed Gemma 2B fine-tune](gemma_sft_pathways_distributed.md) —
  Gemma 2B with LoRA and the Keras `DataParallel` distribution. The
  script sets `backend="pathways"` on a single-host slice to test the
  multi-host code path on one host.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1em` Fine-tuning Gemma 4 on TPU
:link: gemma4_finetuning
:link-type: doc

The complete tutorial: LoRA, saved weights, and inference.
:::

:::{grid-item-card} {octicon}`key;1em` Forward Environment Variables
:link: ../guides/env_vars
:link-type: doc

`capture_env_vars`, the wildcard rules, and how to handle secrets.
:::

:::{grid-item-card} {octicon}`server;1em` Distributed Training
:link: ../guides/distributed_training
:link-type: doc

Multi-host slices, the `pathways` backend, and the return value rules.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: ../guides/checkpointing
:link-type: doc

`KINETIC_OUTPUT_DIR`, retention, and resumable fine-tuning runs.
:::
::::
