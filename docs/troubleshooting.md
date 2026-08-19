# Troubleshooting

This page lists problems by symptom, not by error message. Find the
section that matches what you see, and follow the steps in that section.
The last section lists the checks that the troubleshoot path of
`kinetic init` runs.

:::{admonition} Quick diagnostic
:class: tip

For a fast check of the common environment problems, run:

```bash
kinetic init
```

Select `troubleshoot` at the prompt. The troubleshoot path checks your
local tools, your credentials, your Google Cloud project, and the
cluster infrastructure. It prints a fix hint for each failed check.
`kinetic init` offers this path directly when a prerequisite is
missing. The troubleshoot path diagnoses only a cluster from one of your
profiles. If you have no profile for the project, the troubleshoot path
runs the environment checks only. See
[What the troubleshoot path checks](#what-the-troubleshoot-path-checks).
:::

## Startup and build issues

### "Project must be specified"

Kinetic found no project. No profile is active, and no override supplies
a project. Do one of these two things:

- Run `kinetic init`. Both the **Join** path and the **Create** path save
  a profile and make it active.
- If you already have a profile, run `kinetic profile use NAME`.
  `kinetic profile ls` lists your profiles.

The environment variable `KINETIC_PROJECT` and the decorator argument
`project=` are one-off overrides. Use them for one command or one job,
not as the normal path. `kinetic config` shows the resolved project and
its source. See [Profiles](guides/profiles.md) and
[Configuration](configuration.md).

### "404 Requested entity was not found"

A required Google Cloud resource does not exist. Common examples are the
Artifact Registry repository, a Cloud Storage bucket, or the GKE cluster
itself. Run the setup:

```bash
kinetic up
```

`kinetic up` creates the missing resources and enables the required
APIs. A second `kinetic up` for the same cluster is safe. If the error
remains, run the troubleshoot path of `kinetic init`. The **GCP
Resources** group names the resource that is missing.

If you enable the APIs by hand, enable all seven:

```bash
gcloud services enable compute.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com \
    container.googleapis.com secretmanager.googleapis.com \
    iam.googleapis.com --project <project-id>
```

### The first run takes a long time to start

The first run with a given dependency file builds a container image with
Cloud Build. The build takes about 5 to 10 minutes. Later runs with the
same dependency file reuse the cached image and start in less than 1
minute while a node still runs. Kinetic builds a new image when the
dependency file, your Python minor version, the Kinetic version, or the
accelerator category (CPU, GPU, or TPU) changes. See
[Dependencies](guides/dependencies.md).

If your dependency file changes many times a day, see
[Container Images](guides/containers.md) for a mode that installs the
dependencies at pod start.

### Every submit uploads for a long time

Kinetic logs the size of `context.zip` on every submit. Above 100 MB,
Kinetic logs a warning and lists the five largest files. Read that
warning first. The usual cause is a data directory, a checkpoint
directory, or a virtual environment inside the
[package root](guides/packaging.md). Wrap large inputs in
`kinetic.Data(...)`. Kinetic uploads such an input one time and caches
it by content hash. As an alternative, add the paths to a
`.kineticignore` file.

Kinetic logs a separate warning above 50 MB of *payload*. The payload
holds your arguments, the module-level globals that your function reads,
and the code of your first-party modules. The payload holds no files
from the archive. Load a large object inside the function, or pass it as
`Data`.

You can change both thresholds. Set `KINETIC_CONTEXT_SIZE_WARN_MB` or
`KINETIC_PAYLOAD_SIZE_WARN_MB` to a number of megabytes.

### The container build fails

Kinetic logs the build ID and a `View build:` URL when it submits the
build. Open that URL to read the Cloud Build log. To list the recent
builds of the project, run:

```bash
gcloud builds list --project <project-id> --limit=5
```

Common causes are a package name that does not exist on the index, a
version pin that does not resolve, and a network error during the
install. See [Dependencies](guides/dependencies.md).

## Auth and config issues

### "Permission denied" on Google Cloud operations

Your user account, or your service account, lacks a permission on the
project. `kinetic up` needs broad permissions, because one run creates
many resources:

- A GKE cluster and its node pools.
- Two service accounts and project-level IAM bindings for them.
- A VPC network, a Cloud Router, and a Cloud NAT gateway.
- An Artifact Registry repository and Cloud Storage buckets.
- The required Google Cloud APIs, which `up` enables.

The troubleshoot path reports a `Permission denied` on project access
and recommends `roles/editor` or `roles/owner` on the project. Ask a
project owner for one of these roles before you run `kinetic up`. For
the roles that a teammate needs on the shared state bucket, see
[Configuration](configuration.md#iam).

### Application Default Credentials are missing or expired

Log in again:

```bash
gcloud auth login
gcloud auth application-default login
```

If you set `GOOGLE_APPLICATION_CREDENTIALS` to a service account key
file, that key takes precedence over your user credentials.

### A setting does not take effect

Kinetic resolves each of project, zone, cluster, and namespace in this
order. The first value wins.

1. The decorator argument or the CLI flag.
2. The `KINETIC_*` environment variable.
3. The active profile.
4. The built-in default.

Run `kinetic config`. It shows the resolved value of each setting and its
source: an environment variable, the active profile, or the default.
`kinetic config` cannot see a CLI flag or a decorator argument. If a
value surprises you, look for a `KINETIC_*` variable in your shell,
because an environment variable overrides the profile. See
[Configuration](configuration.md).

## Scheduling and quota issues

### A job stays in `PENDING` for more than 10 minutes

The cluster cannot start a node for the pod. Check these causes in
order:

1. **No node pool matches the accelerator.** A job runs only on a node
   pool with the same accelerator type and, for a TPU, the same
   topology. The cluster does not create accelerator pools automatically.
   Run `kinetic pool list`. If no pool matches, add one:
   `kinetic pool add --accelerator tpu-v5litepod-4` (use your
   accelerator). Kinetic logs an info line at submit when no node
   matches, and then continues.
2. **The Spot setting does not match.** A job with `spot=True` needs a
   pool that you added with `--spot`. A job without `spot=True` needs an
   on-demand pool. See the next section.
3. **No quota for the accelerator** in your zone. Open
   Cloud Console → IAM & Admin → Quotas and filter by your accelerator
   type. Request more quota, or use a different zone.
4. **No Spot capacity.** If the pool uses Spot VMs, Google Cloud can have
   no Spot capacity at that moment. Use an on-demand pool, or try later.

`kubectl describe pod <pod-name>` shows the scheduling events for the
pod. Find the pod with `kubectl get pods -l job-id=<job_id>`. The
troubleshoot path also lists pending pods and reports exhausted
accelerator quotas in your region.

### A `--spot` pool receives no jobs

A pool that you added with `--spot` accepts only jobs that request Spot
capacity. Set `spot=True` on the decorator, or add the `:spot` suffix to
the accelerator string:

```python
@kinetic.run(accelerator="gpu-l4", spot=True)
def train(): ...
```

A job without `spot=True` never runs on a Spot pool. See
[Cost Optimization](guides/cost_optimization.md#spot-vms).

### A multi-host TPU job fails at submit or soon after it starts

Three causes are common:

- A missing LeaderWorkerSet controller. The submit fails with
  `LeaderWorkerSet CRD not found`.
- A topology mismatch. Your code expects a device count that the slice
  does not have.
- One host that fails before the other hosts reach the first collective.

See [Distributed Training](guides/distributed_training.md) for the full
list of multi-host failure modes and their fixes.

## Runtime failures

### `ModuleNotFoundError` / `ImportError` on the pod

Three different causes produce the same exception. Identify your cause
from the module name in the traceback:

| The missing module is | The cause | The fix |
| --------------------- | --------- | ------- |
| A third-party package (`pandas`, `transformers`, and more) | The package is not in your `requirements.txt` or `pyproject.toml`. A local `pip install` has no effect on the pod. | Add the package to the dependency file. Read the `Using dependency file: ...` log line. Kinetic walks up from the entry directory of your function, so Kinetic can select a different file than you expect. See [Dependencies](guides/dependencies.md). |
| One of your own modules or packages (`trainer`, `mylib.utils`, and more) | The module is outside the detected [package root](guides/packaging.md), so it never entered `context.zip`. An exclusion rule can also remove it. | Put a `pyproject.toml`, a `requirements.txt`, or a `.git` at the top of the tree that you want to ship. As an alternative, set `KINETIC_PACKAGE_ROOT`. Check your `.kineticignore` and the default exclusion list. |
| `kinetic` itself | The image does not have Kinetic installed. Your function, or one of your own modules in the payload, references the `kinetic` module, so the pod must import `kinetic` to unpickle the job. An old base image or a custom image can lack it. | Rebuild your image. The image that Kinetic builds installs `keras-kinetic` for you. For a custom image, install `keras-kinetic` in the image. |

The traceback names the frame that failed. A failure inside
`cloudpickle.load` means that the pod needed the module to *deserialize*
your job. A failure inside your function means a runtime `import`, which
resolves against the pod `sys.path`: the workspace first, then the
installed packages.

### `FileNotFoundError` on a path that exists locally

Check these three points, in order:

1. **Is the file inside the [package root](guides/packaging.md)?**
   Kinetic archives the package root only. A file one directory above
   the root never ships.
2. **Did an exclusion rule remove the file?** Kinetic skips `.venv`,
   `node_modules`, the cache directories, and every path that matches
   your `.kineticignore`. Kinetic also excludes a local path that you
   wrap in `Data(...)`. Read such a path through the value that the
   `Data` argument resolves to, not through its original location.
3. **Is the path relative to the directory from which you ran the script?** The
   pod changes to the workspace directory that matches your client
   working directory, so a relative path behaves as it does locally. If
   your client working directory was *outside* the package root, the pod
   uses the workspace root instead.

Do not use an absolute client path to read a shipped file. The runner
does create one symbolic link on the pod. The link is at the path of
your client **entry directory**, which is the directory of the file that
defines the decorated function. The link points at the workspace root,
and it exists so that the debugger can map source files. Use a relative
path.

### Pickle / cloudpickle errors at submit time

Your function, or one of its closures, references an object that
`cloudpickle` cannot serialize. These objects are the usual causes:

- An open file handle.
- A database connection.
- A thread.
- A lock.
- A module-level singleton built for local use.

Move the initialization of these objects inside the decorated function.

Kinetic bisects the payload after a failure and names the component at
fault. One example message is
`kinetic could not serialize argument 2 (type socket): ...`. The message
names the function, one positional argument, or one keyword argument,
instead of a location inside `cloudpickle`. Kinetic counts positional
arguments from `0`.

### Kinetic rejects one of your arguments at submit time

Kinetic rejects three argument shapes at submit time, because none of
them arrives on the pod unchanged:

- A `Data` object inside a `set` or a `frozenset`.
- A `Data` object used as a `dict` key.
- A self-referential structure whose cycle runs through a tuple, a set,
  or a frozenset. Kinetic reports this cycle only when the call also
  holds a `Data` object.

Each message names the position of the argument. See
[What Ships to the Pod](guides/packaging.md) for the full set of
guarantees about argument types.

### Unpickling fails on the pod with a version-skew message

Pickled code objects are not portable across Python minor versions. The
runner compares the client fingerprint in the payload against the pod.
The error that `result()` raises names both sides
(`client Python 3.12.2 / pod Python 3.11.9`). The runner compares the
Python minor version (`X.Y`) only, and the `cloudpickle` version exactly.
The pod log holds a skew warning only when the payload unpickled
correctly.

The image that Kinetic builds always matches the Python of your client.
For a base image that you publish, or for a custom image, you must match
the version yourself. See
[Matching your local environment to the pod](guides/packaging.md#matching-your-local-environment-to-the-pod).

### JAX version mismatch errors

You pinned `jax`, `jaxlib`, or `libtpu` in your dependency file. Kinetic
filters those lines out by default and logs a warning for each one. If
you need a specific version, add `# kn:keep` to the line. You must then
make sure that the JAX version matches the accelerator runtime. See
[Dependencies](guides/dependencies.md).

### The job fails but the logs look normal

The pod exited with a non-zero code and wrote no result payload. The
usual causes are an out-of-memory event (`OOMKilled`) and a signal that
stopped the process. Read the pod events:

```bash
kubectl get pods -l job-id=<job_id>
kubectl describe pod <pod-name>
```

Look for `OOMKilled` or an eviction event. If your profile uses a
namespace other than `default`, add `-n <namespace>` to both commands.

A blocking call, and `result()` on a `JobHandle`, delete the Kubernetes
Job and its pod when they collect the failure. Kubernetes also deletes a
finished Job 10 minutes after it ends (2 hours for a job with
`debug=True`). To inspect the pod, run the job with `run_async()` and
read the events before you call `result()`. As an alternative, call
`result(cleanup=False)`, or run `kinetic jobs result <job_id>
--no-cleanup`, to keep the pod until the 10-minute limit.

On a multi-host job, `kinetic jobs logs` gives you the logs of the
leader pod only. A different host can be the host that failed. Read the
logs of each pod, or use the process index in the local error to find
the host that failed.

### The job does not stop after your function returns

Your function left a non-daemon thread alive: a data-loader worker, a
metrics uploader, or a `ThreadPoolExecutor` that your code did not shut
down. CPython waits for non-daemon threads at interpreter exit. Without
a forced exit, the pod stays alive after the work ends, and you pay for
the accelerator.

Kinetic uploads the result first. Kinetic then logs a warning that names
every thread that is still alive, and forces the process to exit. The
job completes, but the warning reports a real leak. Call `.shutdown()`
or `.join()` on your executors and pools. As an alternative, create your
threads with `daemon=True`.

### The job fails with "function called sys.exit(N)"

Kinetic catches a `sys.exit()` call inside your function. `sys.exit()`
and `sys.exit(0)` report a **success**, and the result is `None`. Any
other exit code fails the job with that message. Return a value instead
of an exit if you want Kinetic to collect a result.

### A job stops after about 24 hours

The node pools that `kinetic up` and `kinetic pool add` create set a
maximum run duration of 24 hours on each VM. Only a Spot TPU pool has no
such limit. Google Cloud stops the VM at that time, the pod ends, and
the job fails. For a job that runs longer than 24 hours, write
checkpoints under `KINETIC_OUTPUT_DIR` and resume the work in a new job.
See [Outputs and Checkpoints](guides/checkpointing.md).

## Missing outputs and results

### "Job failed but no result payload was found"

The pod ended before it wrote anything to
`gs://{jobs bucket}/{job_id}/result.pkl`. Kinetic uploads a failure payload
for each of its own startup phases. That payload carries a `phase`
field, which names the phase at fault:

- `artifact download`
- `artifact verification`
- `requirements install`
- `context extract`
- `payload unpickle`
- `environment setup`
- `data resolve`
- `debugger setup`

If you see this message, the failure happened outside those phases.
These causes remain:

- **Something stopped the container** — an out-of-memory event
  (`OOMKilled`), a node preemption on Spot, or a signal that stopped the
  process. Run
  `kubectl describe pod <pod-name>` and look for `OOMKilled` or an
  eviction event. Find the pod with
  `kubectl get pods -l job-id=<job_id>`.
- **The image cannot start the runner** — a custom image without
  `/app/remote_runner.py`, `python3`, `cloudpickle`,
  `google-cloud-storage`, or `absl-py`. The pod log stops before any
  Kinetic output.
- **Cloud Storage was unreachable** — the runner could not upload the
  failure payload either. The pod log holds the original error.
- **Kinetic already deleted the artifacts** — a blocking call deletes
  the artifacts after it collects a result. For a detached job,
  `handle.result()` deletes the artifacts on success, and `.cleanup()`
  deletes them at any time. A reattach after that point finds nothing.

For a long job, write your artifacts under `KINETIC_OUTPUT_DIR` instead
of a return value. See
[Outputs and Checkpoints](guides/checkpointing.md).

On a multi-host job, only the leader (process 0) writes the result. You
therefore see this message only if no host of the job wrote a record. If
one host wrote a record, Kinetic reports the failure of that host
instead. The next section gives the details.

### A multi-host job fails, but the leader is successful

Only the leader (process 0) writes the result of a multi-host job. Each
other host writes a failure record if it fails. Kinetic reads those
records. Kinetic then raises the exception of the failing host with the
lowest process index.

The local error gives you:

- the exception and the remote traceback of that host
- the process index of that host
- the index of each other host that also failed

Kinetic keeps the Cloud Storage artifacts under
`gs://{bucket}/{job_id}/`, so you can examine the run.

Some failures stop a host before it can write a record. Examples are an
out-of-memory kill, a Spot preemption, and a node eviction. Kinetic then
reports that the result claims success although the job failed, and
gives the exit code of each pod. Run
`kubectl describe pod <pod-name>` and look for `OOMKilled` or an
eviction event.

### Kinetic cannot serialize the return value of your job

Your function returned an object that `cloudpickle` cannot serialize.
The runner then uploads a payload with the flag `serialization_failed`
and a `repr()` of the value. Kinetic reports the job as **failed**,
because the return value is not retrievable, and a success report would
hide the problem. The local error holds the truncated `repr`, so you can
see what your function returned.

Kinetic keeps the Cloud Storage artifacts in this case, so you can
inspect the run afterwards. Kinetic still deletes the Kubernetes
resource, so collect the pod log first if you need it. Return a
serializable value instead, such as a path or a dict of metrics. Write
the heavy object, or the object that holds a handle, to
`KINETIC_OUTPUT_DIR`.

### `result()` cannot unpickle the result after a reattach

The result references classes from your project, or from a package that
this client cannot import. Unpickling needs those types locally. The
client that submitted the job can import them. A reattach from a
different directory, a different virtual environment, or a different
machine can therefore fail.

Kinetic raises a `RuntimeError` that names the job and the artifact URI
(`gs://{jobs bucket}/{job_id}/result.pkl`), and chains the original error.
Kinetic does **not** delete the Cloud Storage artifacts, so you lose
nothing. Reattach from the project directory, with the same packages
installed. As an alternative, download the object and inspect it there.

Return plain data to avoid the problem: a dict, a list, a number, a
string, or an array.

### Files that you wrote inside the job are gone

Two causes are possible:

- You wrote the files under `/tmp` or another pod-local path. Kinetic
  deletes the pod when the job ends, and pod-local files do not survive.
  Write the files under `KINETIC_OUTPUT_DIR`.
- You wrote the files under `KINETIC_OUTPUT_DIR`, and more than 30 days
  have passed. The jobs bucket has a 30-day lifecycle rule. Copy the
  artifacts that you must keep to a bucket without a lifecycle rule. See
  [Outputs and Checkpoints](guides/checkpointing.md) for the retention
  details.

### The log does not stream

A network interruption during a `--follow` stream is the most common
cause. A stream interruption does not affect the pod, because log retrieval is read-only. Run
`kinetic jobs logs <job_id>` without `--follow` for the full log. Run
`kinetic jobs logs <job_id> --tail N` for the last N lines. Both
commands work from any machine.

## What the troubleshoot path checks

The troubleshoot path of `kinetic init` runs eight groups of checks and
prints a fix hint for each check that fails. The groups match the
source in `kinetic/cli/commands/doctor.py`:

1. **Local Tools** — `gcloud`, `kubectl`, and `gke-gcloud-auth-plugin`
   are on your `PATH`.
2. **Authentication** — Application Default Credentials exist and
   refresh, and `gcloud` has an active account.
3. **Configuration** — the troubleshoot path has a project ID, a zone,
   and a cluster name.
4. **GCP Project** — the project exists, you can access it, and billing
   is enabled.
5. **GCP APIs** — the Compute Engine, Cloud Build, Artifact Registry,
   Cloud Storage, Kubernetes Engine, Secret Manager, and IAM APIs are
   enabled.
6. **GCP Resources** — the resources of the cluster exist. These are
   the node and build service accounts, the Artifact Registry
   repository, the jobs and builds buckets, the VPC network, and the
   Cloud NAT gateway.
7. **Infrastructure** — the infrastructure state is present in the state
   bucket, and the GKE cluster is in the `RUNNING` state.
8. **Kubernetes** — the cluster is reachable and healthy. This group
   checks these points:
   - Your `kubeconfig` points at the cluster.
   - The API server responds.
   - The node pools are healthy.
   - The LeaderWorkerSet CRD is installed.
   - The Kinetic Kubernetes service account exists.
   - The NVIDIA drivers are installed when a GPU pool exists.
   - The nodes are healthy.
   - No pod is stuck in `Pending`.
   - The cluster has no warning events such as `FailedScheduling` or
     `OOMKilling`.
   - No accelerator quota in the region is exhausted.

The troubleshoot path diagnoses only a cluster from one of your
profiles. If you have no profile for the project, the troubleshoot path
skips groups 6 to 8. Each failing check prints a fix hint. A hint can
hold more than one command.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1em` Getting Started
:link: getting_started
:link-type: doc

Install Kinetic, run `kinetic init`, and run your first job.
:::

:::{grid-item-card} {octicon}`file-directory;1em` What Ships to the Pod
:link: guides/packaging
:link-type: doc

The packaging contract behind most import failures and most file path
failures.
:::

:::{grid-item-card} {octicon}`book;1em` FAQ
:link: guides/faq
:link-type: doc

Short answers to common questions about how Kinetic works.
:::

:::{grid-item-card} {octicon}`gear;1em` Configuration
:link: configuration
:link-type: doc

Profiles, environment variables, and the precedence rules.
:::
::::
