# Getting Started

This page takes you from an empty machine to your first remote job. You
install Kinetic, run `kinetic init` one time, and run a Keras training
script on a cloud TPU. If your team already has a Kinetic cluster,
`kinetic init` connects you to that cluster.

## Before you start

You need these tools and accounts:

- Python 3.11 or later.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), for the
  install command below. `pip` also works.
- The Google Cloud SDK (`gcloud`): [install guide](https://cloud.google.com/sdk/docs/install).
- `kubectl`: [install guide](https://kubernetes.io/docs/tasks/tools/).
  Kinetic installs the `gke-gcloud-auth-plugin` for you on first use, but
  `kubectl` itself must be on your `PATH`.
- A Google Cloud project with [billing enabled](https://docs.cloud.google.com/billing/docs/how-to/modify-project).

Log in to Google Cloud one time:

```bash
gcloud auth login
gcloud auth application-default login
```

## Step 1: Install Kinetic

```bash
uv pip install keras-kinetic
```

This command installs two things:

- The `kinetic` Python package, with the `@kinetic.run()` decorator.
- The `kinetic` command-line tool, which creates and manages your cloud
  infrastructure.

:::{note}
Kinetic uses [Pulumi](https://www.pulumi.com/) to create cloud resources.
Kinetic downloads the Pulumi CLI to `~/.kinetic/pulumi-cli` on first
use. You do not install Pulumi yourself.
:::

## Step 2: Run `kinetic init`

```bash
kinetic init
```

`kinetic init` checks your local tools and your login, and asks for your
project ID. If the project does not exist, `init` offers to create it and
to link a billing account. `init` then offers one of these paths:

- **Join** — `init` lists the Kinetic clusters that already exist in
  the project (yours or a teammate's), lets you select one, and
  configures `kubectl` for it. Kinetic keeps the infrastructure state in
  a bucket that the whole project shares, so every collaborator sees the
  same list.
- **Create** — `init` runs `kinetic up`. `kinetic up` enables the Google
  Cloud APIs, creates a GKE cluster, adds one **node pool** (a group of
  VMs of one accelerator type) of your choice, and configures `kubectl`.
- **Troubleshoot** — `init` runs diagnostics and prints a fix command for
  each failed check.

`init` asks you which path to take. **Join** is available only when a
cluster exists. When a prerequisite check fails, `init` offers
**Troubleshoot** directly. Both the **Join** path and the **Create** path
end with a saved **profile**, and that profile becomes active.

:::{admonition} What is a profile?
:class: tip

A profile is a saved set of four values: the project, the zone, the
cluster, and the Kubernetes namespace. Kinetic stores profiles in
`~/.kinetic/profiles.json`. Every `kinetic` command and every
`@kinetic.run()` call reads the active profile. You therefore do not
export environment variables and you do not pass `--project` on the
command line. `kinetic profile ls` lists your profiles.
`kinetic profile use NAME` makes a different profile active. See
[Profiles](guides/profiles.md).
:::

When `init` finishes, check the cluster:

```bash
kinetic status
```

## Step 3: Make sure that a node pool matches your job

A job runs only on a node pool with the same accelerator type and, for a
TPU, the same slice shape. The **Create** path adds one node pool during
`kinetic up`. The example script below uses
`accelerator="tpu-v5litepod-1"`. List the pools of your cluster:

```bash
kinetic pool list
```

If the list has no `v5litepod` pool with the `1x1` topology, do one of
these two things:

- Add one: `kinetic pool add --accelerator tpu-v5litepod-1`.
- Change `accelerator=` in the script to an accelerator that the cluster
  has. For example, use `accelerator="tpu-v5litepod-4"` for a `2x2` v5e
  pool, `accelerator="gpu-l4"` for an L4 GPU pool, or
  `accelerator="cpu"` for a run without an accelerator.

See [Accelerators](accelerators.md) for the accelerator names.

## Step 4: Run your first job

Save this script as `fashion_mnist.py`:

```{literalinclude} ../examples/fashion_mnist.py
    :language: python
```

Run the script:

```bash
python fashion_mnist.py
```

Kinetic sends the function to the cluster, streams the log lines to your
terminal, and prints the return value when the job ends.

:::{note}
**Expected time:**

- **First run:** 5 to 10 minutes. Kinetic builds a container image with
  your dependencies through Cloud Build, and the autoscaler starts a TPU
  node. Kinetic caches the image. This script needs no dependency file,
  because the image already contains Keras. Later, a `requirements.txt`
  or a `pyproject.toml` next to your script decides the packages. See
  [Dependencies](guides/dependencies.md).
- **Later runs with the same dependencies:** less than 1 minute while a
  node still runs. Kinetic reuses the cached image and uploads only your
  code. After about 10 idle minutes the node scales down, and the next
  run waits 2 to 5 minutes for a new node.
- **Later runs after a change to the dependencies:** 5 to 10 minutes
  again, because Kinetic builds a new image.
:::

## What happened

Kinetic did five things when you called `train_fashion_mnist()`:

:::{container} kinetic-steps
1. **Package.** Kinetic serialized the function and archived the
   package root, the project directory that holds the script.
2. **Build.** Kinetic built a container image with your dependencies, or
   reused a cached image.
3. **Schedule.** Kinetic created a Kubernetes Job on your cluster. The
   cluster autoscaler started a TPU node for it.
4. **Run.** The pod ran the function and streamed the logs to your
   terminal.
5. **Collect.** The pod uploaded the return value to Cloud Storage.
   Kinetic downloaded the value, returned it, and deleted the job
   resources.
:::

[How Kinetic Works](concepts.md) explains each phase, the vocabulary,
and the choices that you make as your jobs get larger.

## Two habits for every job

- **Use a blocking call while you iterate.** A blocking call, such as
  `train_fashion_mnist()`, blocks until the job ends. When a job runs for
  more than a few minutes, call `train_fashion_mnist.run_async()`
  instead. That call returns a `JobHandle` as soon as Kinetic submitted
  the job. See [Detached Jobs](guides/async_jobs.md).
- **Write files that you want to keep under `KINETIC_OUTPUT_DIR`.** The
  pod sets that environment variable to a Cloud Storage location that
  stays after the pod ends. The pod discards the files under `/tmp` when
  it ends. See [Outputs and Checkpoints](guides/checkpointing.md).

## Clean up

The cluster control plane costs money while the cluster exists, even
when no job runs. When you no longer need the cluster, delete it:

```bash
kinetic down
```

`kinetic down` deletes the cluster, the node pools, the Artifact
Registry repository, and the Cloud Storage buckets of the cluster,
including the job outputs. See [Clusters and Node Pools](guides/clusters.md).

## Next steps

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`light-bulb;1em` How Kinetic Works
:link: concepts
:link-type: doc

The vocabulary and the job lifecycle. Read this page before the guides.
:::

:::{grid-item-card} {octicon}`code-square;1em` Examples
:link: examples
:link-type: doc

Runnable scripts for detached jobs, data, checkpoints, parallel sweeps,
and LLM fine-tuning.
:::

:::{grid-item-card} {octicon}`database;1em` Working with Data
:link: guides/data
:link-type: doc

`kinetic.Data(...)` for inputs. See also
[Outputs and Checkpoints](guides/checkpointing.md) for
`KINETIC_OUTPUT_DIR`.
:::

:::{grid-item-card} {octicon}`clock;1em` Detached Jobs
:link: guides/async_jobs
:link-type: doc

`run_async()`, the job lifecycle, and how to reattach to a job.
:::
::::
