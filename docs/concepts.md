# How Kinetic Works

This page explains the model behind Kinetic. It describes the pieces on
your machine and in your Google Cloud project, what a job is, what
travels with a job, and what comes back. Read this page after your
[first run](getting_started.md) and before the guides. Each section names
the guide that covers the details.

## The one-sentence model

You write a Python function. You decorate the function with
`@kinetic.run(accelerator=...)`. When you call the function, Kinetic runs
the function on that accelerator in your Google Cloud project and returns
the return value to your local process.

```python
import kinetic


@kinetic.run(accelerator="tpu-v5litepod-4")
def train(learning_rate):
  import keras  # heavy imports go inside the function

  ...
  return final_loss


loss = train(0.001)  # runs on a 4-chip TPU v5e slice, returns a float
```

Nothing else about your code changes. Kinetic handles the packaging, the
container image, the cluster, and the transfer of the result.

## The pieces

Kinetic has three parts. Two parts are on your machine, and one part is
in your Google Cloud project.

**On your machine**

- The `kinetic` Python package. It gives you `@kinetic.run()`,
  `kinetic.Data`, and the job APIs.
- The `kinetic` command-line tool. It creates and manages the cloud
  resources, and it inspects jobs.
- The **active profile**. A profile is a saved set of four values:
  project, zone, cluster, and Kubernetes namespace. `kinetic init` creates
  the first profile. Both the package and the command-line tool read the
  active profile, so you do not repeat those values in code or in
  commands. See [Profiles](guides/profiles.md).

**In your Google Cloud project** (created by `kinetic up`, one set per
cluster, plus one state bucket per project)

- A **GKE cluster**. Default name: `kinetic-cluster`.
- One or more **accelerator node pools**. Each node pool holds VMs of
  one accelerator type. A node pool scales to zero when no job runs. A
  job can run only on a node pool with the same accelerator type. The
  cluster also has a small default pool of CPU nodes for system pods and
  for `accelerator="cpu"` jobs. See
  [Clusters and Node Pools](guides/clusters.md).
- An **Artifact Registry repository** for the container images that
  Kinetic builds.
- Two **Cloud Storage buckets**: a jobs bucket for job artifacts and
  outputs, and a builds bucket for Cloud Build. Both buckets delete
  objects that are older than 30 days.
- A **Cloud NAT gateway**, which gives the private cluster nodes access
  to the internet.
- A shared **state bucket**, one per project, where the command-line tool
  keeps the infrastructure state. Teammates with access to this bucket
  see the same clusters.

**Inside a job**

- The **pod**. A Kubernetes pod that runs one container on one node of
  the node pool. The pod downloads your code, runs your function, and
  uploads the result. A multi-host TPU job uses one pod per host.
- The **backend**. Kinetic has two: the `gke` backend runs a single-host
  job as a Kubernetes Job, and the `pathways` backend runs a multi-host
  TPU slice as a LeaderWorkerSet, one pod per host. Kinetic selects the
  backend from the accelerator. See
  [Distributed Training](guides/distributed_training.md).

## The decorator

`@kinetic.run()` takes every setting that a job needs. The parameters
fall into five groups:

| Group | Parameters | Notes |
| ----- | ---------- | ----- |
| Hardware | `accelerator`, `spot`, `backend` | `accelerator` names the hardware and the slice size, for example `"tpu-v5litepod-4"`, `"gpu-l4"`, `"gpu-a100x4"`, or `"cpu"`. `spot=True` requests Spot capacity, and needs a Spot node pool. Kinetic selects `backend` for you. |
| Where | `project`, `zone`, `cluster`, `namespace` | Leave these unset. Kinetic reads them from the active profile. Set one only for a one-off override. |
| Inputs | `volumes`, `capture_env_vars` | `volumes` mounts `kinetic.Data(...)` at fixed paths. `capture_env_vars` copies named local environment variables into the pod. Function arguments are the other input path. |
| Outputs | `output_dir` | The Cloud Storage location that the pod sees as `KINETIC_OUTPUT_DIR`. Defaults to a per-job prefix in the jobs bucket. |
| Advanced | `container_image`, `base_image_repo`, `debug` | `container_image` changes how Kinetic produces the image. `debug=True` attaches a debugger. |

See the [API reference](api.rst) for every parameter and its default.

## Three ways to call a decorated function

| Call | Returns | Use it when |
| ---- | ------- | ----------- |
| `train(...)` — a **blocking call** | The return value of the function | The job is short, or you iterate interactively. The call blocks until the job ends and streams the logs. |
| `train.run_async(...)` — a **detached job** | A `JobHandle`, as soon as Kinetic submitted the job | The job runs for more than a few minutes. You poll status, tail logs, and collect the result later, from any machine. See [Detached Jobs](guides/async_jobs.md). |
| `train.run_async_map(inputs, ...)` — a **batch** | A `BatchHandle`, as soon as Kinetic started the submissions | You run the same function over many inputs, for example a hyperparameter sweep. See [Batched Jobs](guides/batched_jobs.md). |

Submission includes the packaging and, when needed, the image build.

`kinetic.attach(job_id)` rebuilds a `JobHandle` from a job ID such as
`job-3f9a1c2b`, and
`kinetic.list_jobs()` lists the jobs on the cluster. The `kinetic jobs`
command group offers the same operations from the shell.

## What travels with a job

Kinetic uploads three kinds of thing to the jobs bucket for every job,
under `gs://{jobs bucket}/{job_id}/`:

1. **Your function**, in `payload.pkl`. Kinetic serializes the function
   with `cloudpickle`, together with its arguments and the environment
   variables that you capture. Kinetic serializes the modules of your own
   project by value, so the pod does not import them to load the job.
2. **Your project source**, in `context.zip`. Kinetic starts at the
   **entry directory**, the directory of the file that defines the
   decorated function. Kinetic then finds the **package root**: the
   nearest directory at or above the entry directory that holds a
   `pyproject.toml`, a `requirements.txt`, a `setup.py`, a `setup.cfg`,
   or a `.git` entry. Kinetic archives that directory. On the pod, the
   runner extracts the archive into a **workspace**, rebuilds `sys.path`,
   and changes to the workspace directory that matches your working
   directory. Imports and relative paths therefore behave as they do on
   your machine. See [What Ships to the Pod](guides/packaging.md).
3. **Your data**, if you wrap a path in `kinetic.Data(...)`. Kinetic
   uploads local data one time, keyed by a content hash, and gives your
   function a plain filesystem path on the pod. See
   [Working with Data](guides/data.md).

Kinetic reads one more file: your **dependency file**, a
`requirements.txt` or a `pyproject.toml`. That file decides which
packages the image contains. A `pip install` in your local shell does not
carry over. See [Dependencies](guides/dependencies.md).

## The container image

The pod runs a container image. By default, Kinetic builds that image for
you with Cloud Build. The image starts from a Python image that matches
your local Python version. The image contains the accelerator runtime
(JAX with `libtpu` or CUDA), Keras, Kinetic, and the packages from your
dependency file. Kinetic tags the image with a hash of those inputs and
stores it in the Artifact Registry repository of the cluster.

Two consequences follow:

- The first run with a given dependency set takes about 5 to 10 minutes,
  because Cloud Build runs.
- Every later run with the same dependency file skips the build. The
  pod starts in less than 1 minute while a node is still running, or
  after the 2 to 5 minutes that a new node needs. A change to the
  dependency file causes a new build.

Kinetic also has other image modes, for a slow build step or for system
libraries. You do not need them at first. See
[Container Images](guides/containers.md).

## The life of a job

Every call goes through the same five phases:

:::{container} kinetic-steps
1. **Package.** Kinetic resolves the profile, serializes the function,
   archives the package root, uploads the `Data` objects, and uploads
   the artifacts to `gs://{jobs bucket}/{job_id}/`.
2. **Build.** Kinetic reuses a cached image or runs Cloud Build.
3. **Schedule.** Kinetic creates a Kubernetes Job on the cluster, or a
   LeaderWorkerSet for a multi-host TPU slice. The cluster autoscaler
   starts a node in the matching node pool. The job is `PENDING`.
4. **Run.** The pod downloads the artifacts, extracts the source,
   resolves the `Data` paths, sets `KINETIC_OUTPUT_DIR`, and calls your
   function. The job is `RUNNING`. Kinetic streams the pod log to your
   terminal.
5. **Collect.** The pod uploads the return value (or the exception) to
   `gs://{jobs bucket}/{job_id}/result.pkl`. The job is `SUCCEEDED` or
   `FAILED`. Kinetic downloads the value and returns it, or raises the
   exception with the remote traceback.
:::

The pod exits as soon as your function returns. After a blocking call or
a `result()` call collects the result, Kinetic deletes the Kubernetes Job
and, on success, the uploaded artifacts. A detached job that nobody
collects keeps its artifacts until you call `result()` or `cleanup()`, or
until the 30-day rule of the jobs bucket deletes them. Kubernetes deletes
a finished Job resource 10 minutes after it ends, without a call from
you. Kinetic never deletes what you wrote under `KINETIC_OUTPUT_DIR` as
part of job cleanup, but the 30-day rule of the jobs bucket applies.

## Where the results go

A job produces three kinds of output. Each kind has a different channel:

| Output | Channel | Size |
| ------ | ------- | ---- |
| The return value | `result.pkl` in the jobs bucket, then your local process | Small: a metric, a dict, a path |
| Files that you keep | `KINETIC_OUTPUT_DIR`, a per-job Cloud Storage prefix | Any size: exported models, evaluation results |
| Checkpoints | A stable subdirectory under `KINETIC_OUTPUT_DIR`, with `output_dir=` set to a fixed location | Any size; a restart reads them |

Write everything that you want to keep under `KINETIC_OUTPUT_DIR`. The
pod discards its filesystem, including `/tmp`, when it ends. The default
output directory is different for every job. To resume a job from its
checkpoints, set `output_dir=` to a fixed location. See
[Outputs and Checkpoints](guides/checkpointing.md).

## Where the settings come from

Every job needs a project, a zone, a cluster, and a namespace. Kinetic
resolves each value in this order, and the first value wins:

1. The decorator argument (`project=`) or the CLI flag (`--project`).
2. The environment variable (`KINETIC_PROJECT`).
3. The active profile.
4. The built-in default: `us-central1-a`, `kinetic-cluster`, and
   `default`. The project has no default. Without a profile or an
   override, Kinetic raises `Project must be specified`.

The profile is the layer that you set one time. The environment variable
and the flag are for a one-off override.
`kinetic config` prints each resolved value and its source. See
[Configuration](configuration.md).

## What it costs

You pay for accelerator nodes only while a job runs. Each accelerator
node pool scales to zero after about 10 minutes without a job. Three
costs continue while the cluster exists. The first is the GKE control
plane, which Google Cloud covers for one cluster with a monthly credit.
The second is one small `e2-standard-4` node in the default pool. The
third is the Cloud NAT gateway. Cloud Build charges for each
image build. When you no longer need a cluster, `kinetic down` deletes
it. See [Cost Optimization](guides/cost_optimization.md).

## Where to go next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1em` Working with Data
:link: guides/data
:link-type: doc

Ship local files and read Cloud Storage data from your function.
:::

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: guides/checkpointing
:link-type: doc

`KINETIC_OUTPUT_DIR`, retention, and resumable training.
:::

:::{grid-item-card} {octicon}`clock;1em` Detached Jobs
:link: guides/async_jobs
:link-type: doc

Submit, monitor, reattach, and clean up long jobs.
:::

:::{grid-item-card} {octicon}`server;1em` Clusters and Node Pools
:link: guides/clusters
:link-type: doc

Add accelerator pools, share a cluster, and run more than one cluster.
:::
::::
