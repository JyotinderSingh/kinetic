# Examples

```{toctree}
:hidden:

examples/fashion_mnist
examples/simple_demo
examples/example_async_jobs
examples/example_data_api
examples/example_fuse
examples/example_checkpoint
examples/example_keras_checkpoint
examples/example_collections
examples/example_gke
examples/pathways_example
examples/gemma_sft_pathways_distributed
examples/gemma3_sft_demo
examples/tunix_sft
```

This page is a catalog of the runnable example scripts in the
repository. Each example renders on this site and is also a Python file
in the `examples/` directory of the
[GitHub repository](https://github.com/keras-team/kinetic/tree/main/examples).

The examples come in three tiers:

- {bdg-success}`Quickstart` — the first run, with the defaults.
- {bdg-secondary}`Core` — the everyday features: detached jobs, data,
  checkpoints, and parallel sweeps.
- {bdg-secondary}`Advanced` — multi-host Pathways jobs and LLM
  fine-tuning. These need special quota or external credentials.

To run an example, clone the repository, install Kinetic, make sure that
`kinetic init` has saved an active profile, and run the script:

```bash
git clone https://github.com/keras-team/kinetic.git
cd kinetic
uv pip install -e .
kinetic init          # skip this step if you already have an active profile
python examples/fashion_mnist.py
```

Each example names its accelerator in the decorator. Your cluster needs a
node pool with that accelerator, or you change the `accelerator=` value.
See [Getting Started](getting_started.md).

The LLM examples import packages that the default image does not have,
such as `keras-hub`, `tunix`, and `wandb`. Before you run one of them,
put a `requirements.txt` with those packages next to the script, or in
the `examples/` directory. Kinetic reads that file and builds an image
with the packages. See [Dependencies](guides/dependencies.md).

## Quickstart

::::{grid} 1 2 2 3
:gutter: 3
:class-container: sd-text-left

:::{grid-item-card} Fashion-MNIST on a TPU
:link: examples/fashion_mnist.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

The first script to run after `kinetic init`. A small Keras classifier
on Fashion-MNIST that shows that your cluster can schedule a TPU pod and
return a result to your shell.

+++

{bdg-secondary}`Keras` &nbsp;
{bdg-secondary}`TPU`
:::

:::{grid-item-card} Keras + JAX on a CPU node
:link: examples/simple_demo.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

The smallest possible check. Keras on JAX on a CPU node, without
accelerator quota. Use it to test your install before you request
hardware.

+++

{bdg-secondary}`Keras` &nbsp;
{bdg-secondary}`JAX` &nbsp;
{bdg-secondary}`CPU`
:::
::::

## Core

::::{grid} 1 2 2 3
:gutter: 3
:class-container: sd-text-left

:::{grid-item-card} Submit, monitor, and reattach
:link: examples/example_async_jobs.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

The full detached-job API: `run_async()`, `status()`, `tail()`,
`result()`, a reattach from another shell with `kinetic.attach()`, and
`list_jobs()`.

+++

{bdg-secondary}`Async` &nbsp;
{bdg-secondary}`Reattach`
:::

:::{grid-item-card} Ship local files into the job
:link: examples/example_data_api.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Wrap a local directory in `kinetic.Data(...)`. The function receives a
plain filesystem path, and does not know whether the bytes started on
your laptop or in Cloud Storage.

+++

{bdg-secondary}`Data` &nbsp;
{bdg-secondary}`GCS`
:::

:::{grid-item-card} Mount data with FUSE
:link: examples/example_fuse.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

`Data(..., fuse=True)` for volumes, single files, several mounts in one
job, and a mix of mounted and downloaded data.

+++

{bdg-secondary}`Data` &nbsp;
{bdg-secondary}`FUSE`
:::

:::{grid-item-card} Resumable JAX training with Orbax
:link: examples/example_checkpoint.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

JAX training that continues where it stopped. Writes Orbax checkpoints
to `KINETIC_OUTPUT_DIR`, and shows the resume path when you run the same
function again.

+++

{bdg-secondary}`JAX` &nbsp;
{bdg-secondary}`Checkpointing` &nbsp;
{bdg-secondary}`Orbax`
:::

:::{grid-item-card} Resumable Keras training
:link: examples/example_keras_checkpoint.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

The same pattern for Keras. Round-trips `model.get_weights()` through
Orbax, so a restarted job continues at the right step.

+++

{bdg-secondary}`Keras` &nbsp;
{bdg-secondary}`Checkpointing` &nbsp;
{bdg-secondary}`Orbax`
:::

:::{grid-item-card} Parallel hyperparameter sweep
:link: examples/example_collections.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Fan out a grid of jobs with `run_async_map()`, limit the concurrency,
collect the results, and handle the job that fails.

+++

{bdg-secondary}`Sweep` &nbsp;
{bdg-secondary}`Parallel`
:::

:::{grid-item-card} Mix accelerators in one script
:link: examples/example_gke.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

One script that runs work on a CPU pool, a TPU pool, and a GPU pool in
turn. Useful to check which hardware your cluster serves.

+++

{bdg-secondary}`Multi-accelerator` &nbsp;
{bdg-secondary}`Cluster`
:::
::::

## Advanced

::::{grid} 1 2 2 3
:gutter: 3
:class-container: sd-text-left

:::{grid-item-card} Multi-host JAX on Pathways
:link: examples/pathways_example.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

The reference for a slice with more than one TPU host. A short JAX
program that checks that the cross-host collectives work before you
trust them with a real workload.

+++

{bdg-secondary}`JAX` &nbsp;
{bdg-secondary}`Pathways` &nbsp;
{bdg-secondary}`Distributed`
:::

:::{grid-item-card} Distributed Gemma 2B fine-tune
:link: examples/gemma_sft_pathways_distributed.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Supervised fine-tuning of Gemma 2B with LoRA and the Keras
`DataParallel` distribution. Pulls the weights from Kaggle, and forces
the `pathways` backend on a single-host slice to exercise the multi-host
code path.

+++

{bdg-secondary}`LLM` &nbsp;
{bdg-secondary}`Pathways` &nbsp;
{bdg-secondary}`Distributed`
:::

:::{grid-item-card} Single-TPU Gemma 3 fine-tune
:link: examples/gemma3_sft_demo.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Compact Gemma 3 1B fine-tuning on one TPU. A good baseline before you
scale to Pathways, and a worked example of Kaggle credentials in the
pod.

+++

{bdg-secondary}`LLM` &nbsp;
{bdg-secondary}`TPU`
:::

:::{grid-item-card} Tunix SFT
:link: examples/tunix_sft.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Supervised fine-tuning of Gemma 3 with LoRA and QLoRA through Tunix on
a TPU v5e slice, with credentials forwarded through `capture_env_vars`.

+++

{bdg-secondary}`LLM` &nbsp;
{bdg-secondary}`TPU` &nbsp;
{bdg-secondary}`LoRA`
:::
::::

## Tutorials

The pages in the **Examples & Tutorials** section of the sidebar are
longer walkthroughs:

- [Training Keras Models](examples/keras_training.md) — patterns for an
  existing Keras script.
- [Native JAX Training](examples/jax_training.md) — JAX loops,
  single-host parallelism, and multi-host slices.
- [PyTorch Training](examples/pytorch_training.md) — PyTorch on GPU
  nodes.
- [Fine-tuning Gemma 4 on TPU](examples/gemma4_finetuning.md) — a
  complete LoRA fine-tune with inference.
- [Fine-tuning LLMs](examples/llm_finetuning.md) — Keras Hub, Kaggle
  credentials, and LoRA.
- [Running vLLM on TPU](guides/vllm_tpu.md) — vLLM inference on a TPU
  slice.
