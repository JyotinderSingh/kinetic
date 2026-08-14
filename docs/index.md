# Run ML workloads on cloud TPUs and GPUs

```{toctree}
:caption: Start Here
:hidden:

getting_started
guides/execution_modes
troubleshooting
guides/faq
```

```{toctree}
:caption: Core Workflows
:hidden:

guides/data
guides/checkpointing
guides/packaging
guides/dependencies
guides/env_vars
guides/profiles
guides/debugging
guides/profiling
guides/cost_optimization
guides/distributed_training
guides/vllm_tpu
guides/containers
guides/advanced
```

```{toctree}
:caption: Examples & Tutorials
:hidden:

examples/keras_training
examples/jax_training
examples/pytorch_training
examples/gemma4_finetuning
examples/llm_finetuning
examples
```

```{toctree}
:caption: Reference
:hidden:

api
cli
accelerators
configuration
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
Run any Python function on a cloud TPU or GPU with one decorator. No
infrastructure to wire up, no images to build by hand, no multi-host
boilerplate.
:::

::::{container} kinetic-hero-buttons
:::{button-ref} getting_started
:color: primary

Get started
:::

:::{button-ref} examples
:color: secondary

Browse examples
:::
::::

```python
import kinetic


@kinetic.run(accelerator="tpu-v6e-8")
def train_model():
  import keras

  model = keras.Sequential([...])
  model.fit(x_train, y_train)
  return model.history.history["loss"][-1]


final_loss = train_model()  # runs on a TPU v6e-8 slice
```

## Start here

Three entry points cover what most new users need first.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1em` Your first run
:link: getting_started
:link-type: doc

Install, point at a cluster, and run a real Keras job in minutes.
:::

:::{grid-item-card} {octicon}`clock;1em` Long-running jobs
:link: guides/async_jobs
:link-type: doc

Switch from blocking `run()` to detached `run_async()` for jobs that
take hours.
:::

:::{grid-item-card} {octicon}`database;1em` Data and checkpoints
:link: guides/data
:link-type: doc

Ship local files in, write durable artifacts back out via
`KINETIC_OUTPUT_DIR`.
:::
::::

## How Kinetic works

Five short phases on every job.

:::{container} kinetic-steps
1. **Discover.**
   Kinetic captures your function, its package root, and the
   `Data(...)` arguments. Kinetic reads `requirements.txt` or
   `pyproject.toml`. See
   [What Ships to the Pod](guides/packaging.md).

2. **Build or fetch.**
   A container image is produced — built with your dependencies
   (bundled mode) or pulled from a published base (prebuilt mode). See
   [Execution Modes](guides/execution_modes.md).

3. **Schedule.**
   A Kubernetes resource (a `Job` for single-host workloads, a
   `LeaderWorkerSet` for multi-host TPU jobs on the Pathways backend)
   is submitted to your GKE cluster. The autoscaler provisions
   accelerator nodes if needed.

4. **Run.**
   Your function executes inside the pod with `KINETIC_OUTPUT_DIR`
   set; logs stream back to your terminal.

5. **Collect.**
   The return value is serialized to GCS and pulled back to your local
   process. `@kinetic.run()` cleans up the pod and GCS artifacts as
   soon as the result is collected. `run_async()` leaves the pod
   running until you call `.result()` or `.cleanup()` on the returned
   `JobHandle` — important to remember on expensive accelerators.
:::

## Choose your execution mode

Three modes control how dependencies get into the container. See
[Execution Modes](guides/execution_modes.md) for the full
recommendation matrix and per-mode startup expectations.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Bundled
:link: guides/execution_modes
:link-type: doc

Kinetic builds a custom image with your deps baked in. Best for stable
workflows and reproducible runs.

+++
{bdg-success}`default`
:::

:::{grid-item-card} Prebuilt
:link: guides/execution_modes
:link-type: doc

Pulls a published base image, installs your deps at pod startup. Best
for fast iteration when deps change often.
:::

:::{grid-item-card} Custom image
:link: guides/execution_modes
:link-type: doc

Bring your own image URI. Best when you need custom system libraries
or a corporate-vetted base.
:::
::::

## Explore the docs

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`file-directory;1em` Working with data
:link: guides/data
:link-type: doc

Get inputs into the job and durable outputs back out.
:::

:::{grid-item-card} {octicon}`history;1em` Checkpointing
:link: guides/checkpointing
:link-type: doc

Make long jobs resumable with `KINETIC_OUTPUT_DIR`.
:::

:::{grid-item-card} {octicon}`server;1em` Distributed training
:link: guides/distributed_training
:link-type: doc

Scale beyond one host with the Pathways backend.
:::

:::{grid-item-card} {octicon}`graph;1em` Cost optimization
:link: guides/cost_optimization
:link-type: doc

Spot capacity, autoscaling, and cleanup habits that save money.
:::

:::{grid-item-card} {octicon}`bug;1em` Debugging
:link: guides/debugging
:link-type: doc

Interactive debugging and log streaming for remote jobs.
:::

:::{grid-item-card} {octicon}`code-square;1em` Examples
:link: examples
:link-type: doc

Runnable scripts from first run to multi-host LLM fine-tuning.
:::
::::
