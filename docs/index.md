# Run ML workloads on cloud TPUs and GPUs

```{toctree}
:caption: Start Here
:hidden:

getting_started
concepts
troubleshooting
guides/faq
```

```{toctree}
:caption: Run Jobs
:hidden:

guides/data
guides/checkpointing
guides/dependencies
guides/env_vars
guides/async_jobs
guides/batched_jobs
guides/debugging
guides/distributed_training
guides/profiling
```

```{toctree}
:caption: Manage Infrastructure
:hidden:

guides/profiles
guides/clusters
guides/cost_optimization
guides/reservations
```

```{toctree}
:caption: Advanced
:hidden:

guides/containers
guides/packaging
```

```{toctree}
:caption: Examples & Tutorials
:hidden:

examples/keras_training
examples/jax_training
examples/pytorch_training
examples/gemma4_finetuning
examples/llm_finetuning
guides/vllm_tpu
examples
```

```{toctree}
:caption: Reference
:hidden:

api
cli
configuration
accelerators
security
```

```{toctree}
:caption: Contributing
:hidden:

architecture
contributing
code-of-conduct
```

:::{container} kinetic-hero
Run any Python function on a cloud TPU or GPU with one decorator. Kinetic
creates the infrastructure, builds the container image, ships your code,
and returns the result.
:::

::::{container} kinetic-hero-buttons
:::{button-ref} getting_started
:color: primary

Get started
:::

:::{button-ref} concepts
:color: secondary

How it works
:::
::::

```python
import kinetic


@kinetic.run(accelerator="tpu-v5litepod-4")
def train_model():
  import keras

  model = keras.Sequential([...])
  model.fit(x_train, y_train)
  return model.history.history["loss"][-1]


final_loss = train_model()  # runs on a 4-chip TPU v5e slice
```

## Start here

Read these three pages in order. They take about 30 minutes. When
something does not work, see [Troubleshooting](troubleshooting.md), and
for short answers see the [FAQ](guides/faq.md).

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1em` 1. Getting Started
:link: getting_started
:link-type: doc

Install Kinetic, run `kinetic init`, and run a Keras job on a TPU.
:::

:::{grid-item-card} {octicon}`light-bulb;1em` 2. How Kinetic Works
:link: concepts
:link-type: doc

The vocabulary, the job lifecycle, and where your code, data, and
results go.
:::

:::{grid-item-card} {octicon}`code-square;1em` 3. Examples
:link: examples
:link-type: doc

Runnable scripts, from a first run to multi-host LLM fine-tuning.
:::
::::

## What happens on every job

:::{container} kinetic-steps
1. **Package.**
   Kinetic serializes your function and archives your project source.
   `Data(...)` arguments upload one time, keyed by content.

2. **Build.**
   Kinetic builds a container image with the packages from your
   `requirements.txt` or `pyproject.toml`, and caches the image. Later
   runs with the same dependencies skip this step.

3. **Schedule.**
   Kinetic creates a Kubernetes Job on your GKE cluster, or a
   LeaderWorkerSet for a multi-host TPU slice. The autoscaler starts an
   accelerator node in the matching node pool.

4. **Run.**
   The pod runs your function with `KINETIC_OUTPUT_DIR` set. The logs
   stream to your terminal.

5. **Collect.**
   The pod uploads the return value to Cloud Storage. Kinetic downloads
   the value and deletes the job resources. Files that you wrote under
   `KINETIC_OUTPUT_DIR` stay.
:::

## Explore the guides

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`file-directory;1em` Working with Data
:link: guides/data
:link-type: doc

Get inputs into the job with `kinetic.Data(...)`.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: guides/checkpointing
:link-type: doc

Keep files and make long jobs resumable with `KINETIC_OUTPUT_DIR`.
:::

:::{grid-item-card} {octicon}`clock;1em` Detached Jobs
:link: guides/async_jobs
:link-type: doc

`run_async()` for jobs that run more than a few minutes. Reattach from
any machine.
:::

:::{grid-item-card} {octicon}`stack;1em` Profiles
:link: guides/profiles
:link-type: doc

One saved project, zone, cluster, and namespace per cluster. Switch
with one command.
:::

:::{grid-item-card} {octicon}`server;1em` Clusters and Node Pools
:link: guides/clusters
:link-type: doc

Add accelerator pools, share a cluster with a team, and clean up.
:::

:::{grid-item-card} {octicon}`cpu;1em` Distributed Training
:link: guides/distributed_training
:link-type: doc

Scale to a multi-host TPU slice with the Pathways backend.
:::
::::
