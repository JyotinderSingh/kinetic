# Fine-tuning Gemma 4 on TPU

This tutorial fine-tunes
[Gemma 4 Instruct 26B](https://www.kaggle.com/models/keras/gemma4) on
one TPU v5e host with Kinetic. You use Low-Rank Adaptation (LoRA) to
reduce the memory that training needs. You save the adapted weights to
Cloud Storage under `KINETIC_OUTPUT_DIR`. Then you run inference with
the fine-tuned model in a second job. Every step runs from your local
machine.

The model is `gemma4_instruct_26b_a4b`, a Mixture of Experts (MoE)
architecture with 26B total parameters and 4B active parameters per
forward pass. All 26B weights load into memory, about 52 GB in
bfloat16. A `tpu-v5litepod-8` slice (8 chips × 16 GB = 128 GB of HBM)
is the smallest configuration that fits. That slice is one host, so the
job runs as a plain Kubernetes Job on the cluster. The complete script,
with both steps, is
[`examples/gemma4_finetuning/gemma4_finetuning.py`](https://github.com/keras-team/kinetic/blob/main/examples/gemma4_finetuning/gemma4_finetuning.py).

## Before you start

You need these things:

- A Google Cloud project with billing enabled.
- Kinetic installed, and `kinetic init` complete. `kinetic init` creates
  the active profile that supplies the project, the zone, and the
  cluster to every command and every job. See
  [Getting Started](../getting_started.md).
- A `tpu-v5litepod-8` node pool in your cluster. `kinetic pool list`
  shows the pools of the cluster. If no `v5litepod` pool with 8 chips
  exists, add one:
  ```bash
  kinetic pool add --accelerator tpu-v5litepod-8
  ```
- A Kaggle account that has
  [accepted the Gemma 4 terms](https://www.kaggle.com/models/keras/gemma4).
- `KAGGLE_USERNAME` and `KAGGLE_KEY` set in your local shell.

## Check the TPU availability in your zone

Not every zone offers `v5litepod-8` on demand. `us-central1-a` does.
Before you submit the job, make sure that the zone of your profile has
the hardware. `kinetic config` shows the zone and the project of the
active profile. Then run:

```bash
gcloud compute tpus accelerator-types list --zone=us-central1-a --project=your-project-id
```

Look for `v5litepod-8` in the output. If the zone does not list it,
create a profile for a zone that does. See
[Profiles](../guides/profiles.md).

## Forward the Kaggle credentials

The pod downloads the model weights from Kaggle, so the pod needs your
Kaggle credentials. The `capture_env_vars` parameter copies the named
variables from your shell into the pod:

```python
import kinetic


@kinetic.run(
  accelerator="tpu-v5litepod-8",
  capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def fine_tune_gemma4(): ...
```

The pattern `KAGGLE_*` matches `KAGGLE_USERNAME` and `KAGGLE_KEY`. The
pattern `GOOGLE_CLOUD_*` forwards a `GOOGLE_CLOUD_PROJECT` value if your
shell has one. With an active profile, Kinetic does not read
`GOOGLE_CLOUD_*` variables to find the project or the zone; the profile
supplies both.

:::{note}
Kinetic logs a warning when it captures a variable whose name looks like
a credential, such as `KAGGLE_KEY`. The value travels in plaintext inside
the job payload in the jobs bucket. The warning is expected for this
tutorial. See
[Forward Environment Variables](../guides/env_vars.md#secrets).
:::

## Add the dependency file

The image that Kinetic builds does not contain `keras-hub` or the
tokenizer backends. Put a `requirements.txt` next to your script:

```{literalinclude} ../../examples/gemma4_finetuning/requirements.txt
:language: text
```

Kinetic finds the file, builds an image that contains these packages,
and caches the image. A change to the file causes a new build. The
shipped example keeps this file in `examples/gemma4_finetuning/`, next
to the script. See [Dependencies](../guides/dependencies.md).

## Fine-tune with LoRA

The training function loads the model, enables LoRA, and fits the model
on a small instruction-following dataset. The imports live inside the
function, so they run on the pod.

Four decisions in the code need an explanation:

**Precision policy.** The 26B model stores about 52 GB of weights. The
`mixed_bfloat16` policy keeps float32 master copies of every variable
(about 13 GB per chip on 8 chips). Those copies, together with the MoE
activation tensors, exceed the HBM of one chip. The `bfloat16` policy
stores the variables directly as bfloat16 (about 6.5 GB per chip), which
fits.

**Sequence length.** The MoE activation tensors scale with the compiled
sequence length. The preset default of about 1024 tokens produces about
10 GB per chip of temporary HLO buffers. The line
`model.preprocessor.sequence_length = 128`, before `compile()`, keeps
those buffers under about 2 GB per chip.

**Weight sharding.** The 26B model does not fit on one 16 GB chip.
`ModelParallel` with an explicit `LayoutMap` splits the weights across
all 8 chips when Keras creates the variables. Set the `LayoutMap` before
you call `from_preset()`, so that every variable gets the correct
sharding from the start.

**Custom weight loading.** The Kaggle preset stores the weights in 6
sharded H5 files that a `model.weights.json` manifest describes. The
built-in `load_weights()` on the full `CausalLM` adds a `backbone/`
prefix that matches no path in the manifest.
`model.backbone.load_weights()` avoids that prefix, but the Keras
`ShardedH5IOStore` has a bug. After the store switches to a different
shard file, it does not update its internal `current_shard_path`
pointer. A later `keys()` call restores the stale path. Every layer
whose weights span more than one shard then fails to load with a
"received 0 variables" error. Each MoE expert bank and the token
embedding are such layers. The example therefore bypasses
`ShardedH5IOStore` and reads the H5 files directly with `h5py`. The
loader shards each tensor with `jax.device_put` before it assigns the
tensor, which avoids a memory spike on device 0. The function
`_load_sharded_weights()` in the
[example script](https://github.com/keras-team/kinetic/blob/main/examples/gemma4_finetuning/gemma4_finetuning.py)
contains the complete loader.

:::{note}
`_load_sharded_weights()` is a workaround. When Keras offers a public
loading path that handles the `backbone/` prefix and that switches
shards correctly, the loader is no longer necessary.
:::

The code below expects `_load_sharded_weights` and `_make_layout_map`
as the example script defines them.

```python
import os
import kinetic


def _make_layout_map(keras):
  """Build the ModelParallel layout map for Gemma4 26B-A4B."""
  import numpy as np

  devices = keras.distribution.list_devices()
  mesh = keras.distribution.DeviceMesh(
    shape=(1, len(devices)),
    axis_names=["batch", "model"],
    devices=np.array(devices).reshape(1, len(devices)),
  )
  layout_map = keras.distribution.LayoutMap(mesh)
  layout_map[".*moe_expert_bank/gate_proj"] = (None, None, "model")
  layout_map[".*moe_expert_bank/up_proj"] = (None, None, "model")
  layout_map[".*moe_expert_bank/down_proj"] = (None, None, "model")
  layout_map[".*query/kernel"] = ("model", None, None)
  layout_map[".*key/kernel"] = (None, "model", None)
  layout_map[".*value/kernel"] = (None, "model", None)
  layout_map[".*attention_output/kernel"] = ("model", None, None)
  layout_map[".*ffw_gating/kernel"] = (None, "model")
  layout_map[".*ffw_gating_2/kernel"] = (None, "model")
  layout_map[".*ffw_linear/kernel"] = ("model", None)
  layout_map[".*per_layer_input_gate/kernel"] = (None, "model")
  layout_map[".*per_layer_up_proj/kernel"] = (None, "model")
  layout_map[".*token_embedding/embeddings"] = ("model", None)
  keras.distribution.set_distribution(
    keras.distribution.ModelParallel(
      layout_map=layout_map, batch_dim_name="batch"
    )
  )


@kinetic.run(
  accelerator="tpu-v5litepod-8",
  capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def fine_tune_gemma4():
  import h5py
  import io

  import jax
  import keras
  import keras_hub
  import kagglehub
  import numpy as np

  prompts = [
    "<start_of_turn>user\nExplain what a transformer is in one paragraph.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWrite a Python function that reverses a string.<end_of_turn>\n<start_of_turn>model\n",
    # ... more examples
  ]
  responses = [
    "A transformer is a neural network architecture...",
    "def reverse_string(s: str) -> str:\n    return s[::-1]",
    # ...
  ]

  keras.mixed_precision.set_global_policy("bfloat16")
  _make_layout_map(keras)

  print(
    "Loading Gemma 4 Instruct 26B weights (~52 GB, this may take several minutes)..."
  )
  model = keras_hub.models.Gemma4CausalLM.from_preset(
    "gemma4_instruct_26b_a4b",
    load_weights=False,
  )
  model_path = kagglehub.model_download(
    "keras/gemma4/keras/gemma4_instruct_26b_a4b"
  )
  _load_sharded_weights(
    model.backbone, os.path.join(model_path, "model.weights.json")
  )

  model.backbone.enable_lora(rank=4)
  print(f"Trainable parameters: {model.count_params():,}")

  model.preprocessor.sequence_length = 128
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5))
  model.fit(
    x={"prompts": prompts, "responses": responses}, epochs=1, batch_size=1
  )

  output_dir = os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/gemma4_lora")
  weights_path = f"{output_dir}/gemma4_lora.weights.h5"

  buffer = io.BytesIO()
  with h5py.File(buffer, "w") as f:
    for var in model.trainable_variables:
      val = np.asarray(jax.device_get(var.value), dtype=np.float32)
      f.create_dataset(var.path, data=val)

  if weights_path.startswith("gs://"):
    from google.cloud import storage as gcs_storage

    without_scheme = weights_path[5:]
    bucket_name, _, blob_name = without_scheme.partition("/")
    blob = gcs_storage.Client().bucket(bucket_name).blob(blob_name)
    buffer.seek(0)
    blob.upload_from_file(buffer, content_type="application/x-hdf5")
  else:
    os.makedirs(output_dir, exist_ok=True)
    with open(weights_path, "wb") as out_f:
      out_f.write(buffer.getvalue())

  print(f"LoRA weights saved to: {weights_path}")
  return weights_path


if __name__ == "__main__":
  os.environ["KERAS_BACKEND"] = "jax"

  weights_path = fine_tune_gemma4()
  print(f"Training complete. Weights at: {weights_path}")
```

The function saves only the LoRA adapter variables, a few MB, and not
the full 26B backbone. Kinetic sets `KINETIC_OUTPUT_DIR` in the pod to a
Cloud Storage prefix that is unique to the job. The default is
`gs://{project}-kn-{cluster}-jobs/outputs/{job_id}`, for example
`gs://your-project-id-kn-kinetic-cluster-jobs/outputs/job-534ffeb6`.
The function prints the full path of the weights file, and you pass that
path to the inference job below.

The shipped script also sets `GOOGLE_CLOUD_PROJECT`, `KINETIC_ZONE`, and
`GOOGLE_CLOUD_ZONE` in its `__main__` block. With an active profile,
Kinetic ignores the two `GOOGLE_CLOUD_*` variables; their only effect
is that `capture_env_vars` forwards them to the pod. `KINETIC_ZONE`
overrides the zone of the active profile. Set the three values to match
your profile, or delete the three lines.

:::{warning}
A lifecycle rule on the jobs bucket deletes every object 30 days after
its creation. If you need the LoRA weights for longer, copy the file to
a bucket that you manage, or pass `output_dir="gs://your-bucket/gemma4"`
to `@kinetic.run()`. See
[Outputs and Checkpoints](../guides/checkpointing.md#retention-and-cleanup).
:::

A blocking call to `fine_tune_gemma4()` blocks until the job ends. The
first run builds the container image (5 to 10 minutes), and every run
downloads about 52 GB of weights from Kaggle before training starts.
If you do not want to keep a terminal open, call
`fine_tune_gemma4.run_async()` and collect the result later. See
[Detached Jobs](../guides/async_jobs.md).

## Monitor the job

While the job runs, inspect it from a second terminal with the
`kinetic jobs` commands. The active profile supplies the project and the
cluster. Add `--project` or `--cluster` only when the job runs on a
different cluster.

List the live jobs:

```bash
kinetic jobs list
```

Show the status of one job. Kinetic prints the job ID, for example
`job-534ffeb6`, in the log output when it submits the job:

```bash
kinetic jobs status JOB_ID
```

Stream the logs until the job ends:

```bash
kinetic jobs logs --follow JOB_ID
```

Stop the job early:

```bash
kinetic jobs cancel JOB_ID
```

If the job stays in `PENDING` for more than 10 minutes, inspect the
pod. A `tpu-v5litepod-8` job is a plain Kubernetes Job named
`kinetic-JOB_ID`, and its pod carries the label `job-id=JOB_ID`:

```bash
kubectl get pods -l job-id=JOB_ID -n default
kubectl describe pod -l job-name=kinetic-JOB_ID -n default
```

Replace `default` with the namespace of your profile if you changed it.
Read the **Events** section at the end of the output. The common causes
are insufficient TPU quota, no node pool that matches the accelerator,
and an image pull error. See
[Scheduling and quota issues](../troubleshooting.md#scheduling-and-quota-issues)
in Troubleshooting.

## Run inference with the fine-tuned weights

After the training job ends, copy the printed weights path and pass it
to a second job:

```python
import os
import kinetic


@kinetic.run(
  accelerator="tpu-v5litepod-8",
  capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def run_inference(weights_path: str):
  import h5py
  import io

  import keras
  import keras_hub
  import kagglehub
  import numpy as np

  keras.mixed_precision.set_global_policy("bfloat16")
  _make_layout_map(keras)

  print("Loading Gemma 4 Instruct 26B weights (~52 GB)...")
  model = keras_hub.models.Gemma4CausalLM.from_preset(
    "gemma4_instruct_26b_a4b",
    load_weights=False,
  )
  model_path = kagglehub.model_download(
    "keras/gemma4/keras/gemma4_instruct_26b_a4b"
  )
  _load_sharded_weights(
    model.backbone, os.path.join(model_path, "model.weights.json")
  )

  model.backbone.enable_lora(rank=4)
  print(f"Loading LoRA weights from: {weights_path}")

  if weights_path.startswith("gs://"):
    from google.cloud import storage as gcs_storage

    without_scheme = weights_path[5:]
    bucket_name, _, blob_name = without_scheme.partition("/")
    buffer = io.BytesIO()
    gcs_storage.Client().bucket(bucket_name).blob(blob_name).download_to_file(
      buffer
    )
    buffer.seek(0)
    h5_source = buffer
  else:
    h5_source = weights_path

  path_to_var = {var.path: var for var in model.trainable_variables}
  with h5py.File(h5_source, "r") as f:
    for path, var in path_to_var.items():
      if path in f:
        var.assign(np.array(f[path]))

  prompt = (
    "<start_of_turn>user\n"
    "Explain what a transformer is in one paragraph."
    "<end_of_turn>\n<start_of_turn>model\n"
  )
  output = model.generate([prompt], max_length=256)
  return output[0]


if __name__ == "__main__":
  os.environ["KERAS_BACKEND"] = "jax"

  # Replace with the path that the fine-tuning job printed.
  weights_path = "gs://your-project-id-kn-kinetic-cluster-jobs/outputs/job-534ffeb6/gemma4_lora.weights.h5"
  response = run_inference(weights_path)
  print(response)
```

The inference job loads the base weights again from Kaggle, enables LoRA
with the same rank, and assigns the saved adapter variables by path. It
then generates one response and returns the text.

## Clean up

A node pool that you add with the default `--min-nodes 0` scales to zero
after about 10 idle minutes. After that, the TPU nodes cost nothing. The
cluster control plane and the default CPU node cost money while the
cluster exists.

To delete the `v5litepod-8` pool and keep the cluster for other jobs:

```bash
# Find the exact pool name.
kinetic pool list

# Delete the pool. Use the name from the list, for example tpu-v5litepod-a1b2.
kinetic pool remove POOL_NAME
```

To delete the whole cluster, with every pool and the buckets:

```bash
kinetic down
```

:::{warning}
`kinetic down` deletes the jobs bucket. The LoRA weights under the
default `KINETIC_OUTPUT_DIR` are in that bucket. Copy the weights to a
bucket that Kinetic does not manage before you run `kinetic down`.
:::

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book;1em` Fine-tuning LLMs
:link: llm_finetuning
:link-type: doc

The general patterns for Keras Hub models: the dependency file, the
Kaggle credentials, and the move to a multi-host slice.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: ../guides/checkpointing
:link-type: doc

`KINETIC_OUTPUT_DIR`, retention, and how to write checkpoints that a
long run can resume from.
:::

:::{grid-item-card} {octicon}`key;1em` Forward Environment Variables
:link: ../guides/env_vars
:link-type: doc

How `capture_env_vars` works, which names a wildcard never matches, and
how Kinetic handles secrets.
:::

:::{grid-item-card} {octicon}`server;1em` Distributed Training
:link: ../guides/distributed_training
:link-type: doc

Move to a larger TPU slice that spans more than one host.
:::
::::
