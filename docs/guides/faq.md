# FAQ

Short answers to the questions that new users ask most. Each answer
links to the page with the details. The [glossary](#glossary) at the end
defines the terms that the documentation uses.

## How do I tell Kinetic which project and cluster to use?

Run `kinetic init` one time. It saves a **profile** with your project,
zone, cluster, and namespace, and makes that profile active. Every
`kinetic` command and every `@kinetic.run()` call reads the active
profile. To work with a second cluster, create a second profile and
switch with `kinetic profile use NAME`. For a one-off override, pass
`project=` or `cluster=` to the decorator, or `--project` and
`--cluster` to a command. See [Profiles](profiles.md) and
[Configuration](../configuration.md).

## When do I call the function directly, and when do I use `run_async()`?

Call the function directly (a **blocking call**) when you want your
script to wait for the result. The call blocks and streams the logs. Use
`func.run_async()` when the job runs for more than a few minutes. Use it
also when you want to leave your machine, or when you want to watch
several jobs at the same time. `run_async()` returns a `JobHandle` as
soon as Kinetic submitted the job. See [Detached Jobs](async_jobs.md).

## Why is the first run slow?

The first run with a given dependency file builds a container image with
Cloud Build. The build takes about 5 to 10 minutes. Kinetic tags the
image with a hash of the dependency file, the accelerator category, the
Python version, and the Kinetic version. Every later run with the same
inputs reuses the image and starts in less than 1 minute. A change to the
dependency file causes a new build. If you change the dependency file
many times a day, see [Container Images](containers.md) for the prebuilt
mode. See also [Dependencies](dependencies.md).

## Which packages does the pod have?

The image contains JAX for your accelerator, Keras, Kinetic, and the
packages in the `requirements.txt` or `pyproject.toml` that Kinetic finds
next to your script or above it. A `pip install` in your local shell
does not carry over. Kinetic logs the file that it selected on each
submit. See [Dependencies](dependencies.md).

## Which directory does Kinetic upload with my job?

Kinetic uploads the *package root*. Kinetic starts at the file that
defines your decorated function. Kinetic walks up out of each directory
that holds an `__init__.py` file. Kinetic then walks up to the nearest
directory that holds a `pyproject.toml`, a `requirements.txt`, a
`setup.py`, a `setup.cfg`, or a `.git` entry. The walk stops at your
home directory and at the root of the file system.

Kinetic zips that directory, without these paths:

- The paths in your `Data(...)` objects.
- The default exclusion list: `.venv`, `node_modules`, and the cache
  directories.
- Each path that a `.kineticignore` pattern matches.

On the pod, the runner rebuilds `sys.path` and the working directory to
match your client. Your imports and your relative paths therefore work
as they do locally. To select the directory yourself, set
`KINETIC_PACKAGE_ROOT`. A notebook and a REPL have no source file, so
Kinetic uses your current working directory. See
[What Ships to the Pod](packaging.md).

## When do I use `Data(...)`, and when do I use a `gs://` URI directly?

Use `kinetic.Data(...)`. It accepts a local path and a `gs://` URI, and
it resolves to a plain filesystem path on the pod. Your function sees a
path in every case: a local directory that Kinetic uploads, a bucket
that already exists, or a FUSE mount with `Data(..., fuse=True)`. Read a
`gs://` URI directly in your code only when you have a specific reason to
bypass `Data`. See [Working with Data](data.md).

## How do I keep checkpoints and other files?

Write them under `KINETIC_OUTPUT_DIR`. Kinetic sets that environment
variable in the pod to a per-job Cloud Storage location. Files under
that location outlive the pod, and you can read them from your machine.
The return value of the function is for small results. Files and
checkpoints belong in the output directory. See
[Outputs and Checkpoints](checkpointing.md).

## How do I reattach to a job?

Call `kinetic.attach(job_id)`. It rebuilds a `JobHandle` from the
metadata that Kinetic stored in Cloud Storage at submit time. You can
then call `.status()`, `.result()`, `.tail()`, or `.cleanup()` from any
machine that has Kinetic and your Google Cloud credentials. The job ID
is on the `JobHandle` that `run_async()` returned. If you lost the ID,
`kinetic.list_jobs()` or `kinetic jobs list` lists the jobs on the
cluster. See [Detached Jobs](async_jobs.md).

One condition: `.result()` unpickles the return value on your machine.
If the job returned an instance of one of your own classes, the machine
that reattaches must be able to import that class. Reattach from the
project directory, or return plain data.

## What does Kinetic clean up automatically?

A blocking call and a `JobHandle.result()` call delete the Kubernetes
Job and its pod, for a job that succeeded and for a job that failed. A
job with `debug=True` keeps its Kubernetes resources until the 2-hour
Kubernetes limit. To keep the resources for any other job, call
`handle.result(cleanup=False)`, read the logs, and then call
`handle.cleanup()`.

Kinetic deletes the Cloud Storage artifacts of a job (the uploaded code,
the requirements, and the metadata) only when it collects a usable
result. Kinetic keeps the artifacts when the job failed, when the job
wrote no result, when the pod could not serialize the return value, or
when your client could not deserialize it. For a job that you never
collect, call `JobHandle.cleanup(gcs=True)`.

Kinetic never deletes what you wrote under `KINETIC_OUTPUT_DIR` as part
of job cleanup. The default output directory is in the jobs bucket, and
that bucket deletes objects after 30 days. To keep outputs longer, copy
them to another bucket, or set `output_dir` to a location outside the
jobs bucket. `kinetic down` deletes the jobs bucket with everything in it.

## How do Spot VMs affect training?

Spot capacity costs much less than on-demand capacity, but Google Cloud
can preempt a Spot VM with 30 seconds of notice. A single-host job with
frequent checkpoints recovers well. A multi-host TPU slice does not,
because the loss of one host fails the whole slice. Use `--spot` on a
node pool for fault-tolerant single-host work, and write checkpoints
often enough to absorb a restart. See
[Cost Optimization](cost_optimization.md).

## When do I need more than one cluster?

Most people do not. Create a second cluster to separate GPU and TPU
work, to run in a second region, or to separate development from
production. Each cluster has its own control plane cost. See
[Clusters and Node Pools](clusters.md).

## What is Pathways, and when does Kinetic use it?

[Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)
is a JAX runtime that coordinates execution across many TPU hosts.
Kinetic selects the Pathways backend automatically when the accelerator
spans more than one host, for example `tpu-v5litepod-16` or
`tpu-v6e-16`. JAX collectives and the Keras distribution API then work
across hosts, and you write no coordination code. You pass
`backend="pathways"` yourself only to test the Pathways code path on a
single-host slice. See [Distributed Training](distributed_training.md)
and the **Hosts** column in [Accelerators](../accelerators.md).

## Glossary

**Accelerator** — The string that you pass to `accelerator=`, for example `tpu-v5litepod-8`, `gpu-l4`, `gpu-a100x4`, or `cpu`. It names the hardware and, for a TPU, the slice size.

**Topology** — The arrangement of TPU chips in a slice, for example `2x4`. Kinetic derives it from the accelerator name. A topology that spans more than one host makes the job multi-host.

**Pathways** — The JAX runtime that Kinetic uses for multi-host TPU slices. Kinetic selects it automatically.

**Profile** — A saved set of project, zone, cluster, and namespace. One profile is active at a time. `kinetic init` creates the first one.

**Cluster** — A GKE cluster with its own image repository and buckets. Default name `kinetic-cluster`. Created by `kinetic up`, deleted by `kinetic down`.

**Node pool** — A group of VMs of one accelerator type inside a cluster. Created by `kinetic pool add`. Scales between `--min-nodes` (default 0) and a fixed maximum: 10 more nodes for a GPU pool, or the hosts of one slice for a TPU pool.

**Job** — One execution of a decorated function on the cluster. A job has an ID such as `job-3f9a1c2b`.

**Handle** — A `JobHandle` for one detached job, or a `BatchHandle` for a batch. Wraps `status()`, `result()`, `tail()`, `cancel()`, and `cleanup()`.

**Entry directory** — The directory of the file that defines the decorated function. Kinetic starts its search for the package root and for the dependency file there.

**Package root** — The directory that Kinetic archives and ships with the job, in `context.zip`.

**Workspace** — The directory on the pod into which the runner extracts the package root.

**Dependency file** — The `requirements.txt` or `pyproject.toml` that decides which packages the image contains.

**Image** — The container image that the pod runs. Kinetic builds it by default. See [Container Images](containers.md) for the other modes.

**Output directory** — The Cloud Storage location in `KINETIC_OUTPUT_DIR`. The place for checkpoints and every file that you want to keep.

**FUSE** — A mount of a Cloud Storage location into the pod filesystem. With `Data(..., fuse=True)`, the pod reads files on demand instead of a download at start.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`light-bulb;1em` How Kinetic Works
:link: ../concepts
:link-type: doc

The model behind every answer on this page.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

Symptom-first diagnosis.
:::

:::{grid-item-card} {octicon}`rocket;1em` Getting Started
:link: ../getting_started
:link-type: doc

Your first run, end to end.
:::
::::
