# Running vLLM on TPU

This page shows how to run [vLLM](https://docs.vllm.ai/) inference on a
Cloud TPU with Kinetic. You add one dependency file, forward one
environment variable, and run the example script. Read this page if you
want to generate text with a large language model, such as Llama 3.1, on
a TPU slice.

## Before you start

You need these things:

- A Kinetic cluster and an active profile. `kinetic init` creates both.
  See [Getting Started](../getting_started.md).
- A node pool that matches the accelerator of the example. The example
  uses `accelerator="tpu-v5litepod"`, a 4-chip TPU v5e slice on one host.
  Run `kinetic pool list`. If the list has no matching pool, add one:

  ```bash
  kinetic pool add --accelerator tpu-v5litepod-4
  ```

- A Hugging Face token if the model is gated. The example uses
  `meta-llama/Llama-3.1-8B`, which is a gated model. Request access on
  the model page. Then create a token in your Hugging Face account
  settings.

## Step 1: Add `vllm-tpu` to a dependency file

Save a file with the name `requirements.txt` in the same directory as
your script:

```text
vllm-tpu
```

When you call the decorated function, Kinetic looks for a dependency
file. Kinetic starts the search in the directory of the script and
continues in the parent directories. A `requirements.txt` next to the
script is therefore the first file that Kinetic finds. See
[Dependencies](dependencies.md) for the search rules.

Kinetic then builds a container image that contains `vllm-tpu` and
caches the image. The first run waits for that build. Later runs with
an unchanged file reuse the image.

:::{note}
Do not add `jax`, `jaxlib`, or `libtpu` to the file. The image that
Kinetic builds already contains JAX with the TPU runtime. If the file
has such a line, Kinetic removes the line and logs a warning. See
[JAX and accelerator runtimes](dependencies.md#jax-and-accelerator-runtimes).
:::

## Step 2: Review the environment variables

The example uses two kinds of environment variable. The vLLM variables
are constants, and the script sets them inside the function. The
Hugging Face token is a secret, and the script forwards it from your
shell. You do not set anything in this step. Step 4 sets the token.

### The vLLM variables

The function sets three variables with `os.environ` before the
`from vllm import LLM, SamplingParams` line. The values therefore exist
in the pod process when vLLM loads.

| Variable | Value in the example | Purpose |
| -------- | -------------------- | ------- |
| `VLLM_TARGET_DEVICE` | `tpu` | Selects the TPU backend of vLLM. |
| `JAX_PLATFORMS` | `tpu,cpu` | Lists the JAX backends. Kinetic sets `JAX_PLATFORMS=tpu` in every TPU pod. The example replaces that value with `tpu,cpu`. |
| `VLLM_USE_V1` | `0` | Selects the vLLM engine version. |

You do not set these variables in your shell, and you do not forward
them with `capture_env_vars`. The code carries them.

### The Hugging Face token

One value comes from your shell: `HF_TOKEN`. The decorator lists the
name in `capture_env_vars=["HF_TOKEN"]`. When you call the function,
Kinetic reads `HF_TOKEN` from your environment and copies the value into
the job payload. The pod applies the value before the pod calls your
function, so vLLM can download a gated model. If your shell has no
`HF_TOKEN`, Kinetic captures nothing and logs no error. The pod then
has no token, and the download of a gated model fails. See
[Forward Environment Variables](env_vars.md).

:::{warning}
The name `HF_TOKEN` contains `TOKEN`, so Kinetic logs a warning when it
captures the value. Kinetic stores the value in plaintext inside the job
payload in the jobs bucket, and every job pod in the cluster can read
that bucket. Kinetic deletes the payload when a blocking call or a
`result()` call collects a successful result with the default cleanup.
Otherwise the payload stays until the 30-day lifecycle rule of the
bucket deletes it. If you intend to forward the token, the warning needs
no action. See [Secrets](env_vars.md#secrets) and
[Security](../security.md).
:::

## Step 3: Read the example

Save this script as `vllm_demo.py`, next to the `requirements.txt` from
step 1:

```{literalinclude} ../../examples/vllm_demo.py
:language: python
```

Four points in the script matter:

- **The accelerator.** `accelerator="tpu-v5litepod"` resolves to the
  default v5e slice: 4 chips on one host, topology `2x2`. The string
  `tpu-v5litepod-4` names the same slice. See
  [Accelerators](../accelerators.md).
- **The parallelism.** `tensor_parallel_size=4` matches the 4 chips of
  the slice.
- **The import.** The `from vllm import ...` line is inside the
  function. The pod runs that line, and the image on the pod contains
  vLLM. Your machine does not need vLLM.
- **The result.** The function prints the completions in the pod, and
  Kinetic streams the pod log to your terminal. If you want the
  completions in your local process, return them from the function.

## Step 4: Run the script

Set `HF_TOKEN` for the command and run the script:

```bash
HF_TOKEN=your-hf-token python vllm_demo.py
```

The active profile supplies the project, the zone, and the cluster. You
do not pass them. Kinetic then does these things:

:::{container} kinetic-steps
1. **Package.** Kinetic captures `HF_TOKEN`, serializes the function, and
   archives the directory of the script.
2. **Build.** Kinetic builds an image with `vllm-tpu` on the first run,
   or reuses the cached image.
3. **Schedule.** The cluster autoscaler starts a TPU v5e node in the
   matching node pool if no free node exists.
4. **Run.** The pod downloads the model weights from Hugging Face, loads
   the model on the TPU, and prints the completions. Kinetic streams the
   log lines to your terminal.
5. **Collect.** The call returns when the function ends. Kinetic deletes
   the job resources.
:::

:::{note}
The first run waits for the image build. Every run waits for a node
start if the node pool has scaled to zero. Every run also downloads the
model weights again, because the pod filesystem does not persist between
jobs. If you expect a long run, call `run_vllm_inference.run_async()`
instead of the blocking call, and collect the result later. See
[Detached Jobs](async_jobs.md).
:::

## Change the model or the slice

- **The model.** Change `model_id` to another Hugging Face model. If the
  model is gated, your token must have access to the model.
- **The slice.** Change `accelerator` to another single-host slice, for
  example `tpu-v5litepod-8` (8 chips on one host). Set
  `tensor_parallel_size` to the number of chips in the slice. Add a node
  pool that matches the new accelerator with `kinetic pool add`.
- **Larger slices.** A v5e slice with 16 or more chips spans more than
  one host. Kinetic runs such a job on the Pathways backend, one pod per
  host. See [Distributed Training](distributed_training.md).

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`key;1em` Forward Environment Variables
:link: env_vars
:link-type: doc

`capture_env_vars`, wildcards, and how Kinetic handles a secret such as
`HF_TOKEN`.
:::

:::{grid-item-card} {octicon}`package;1em` Dependencies
:link: dependencies
:link-type: doc

How Kinetic finds `requirements.txt`, and which lines Kinetic filters.
:::

:::{grid-item-card} {octicon}`cpu;1em` Accelerators
:link: ../accelerators
:link-type: doc

Every accelerator name, and the topology behind each TPU name.
:::
::::
