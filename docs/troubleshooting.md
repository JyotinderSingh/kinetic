# Troubleshooting

This page is organized by symptom rather than by error message. Locate
the section that best matches what you are observing and follow the
guidance there.

:::{admonition} Quick diagnostic
:class: tip

For a quick diagnostic of common environment problems, run:

```bash
kinetic init
```

and choose `troubleshoot` at the prompt. It checks for missing tools,
misconfigured credentials, and unhealthy infrastructure, and prints a
concrete fix command for each failed check. `kinetic init` will also
offer this path automatically when it detects that prerequisites are
missing. The full list of categories the troubleshoot path covers is
described at the end of this page.
:::

## Startup and build issues

### "Project must be specified"

`KINETIC_PROJECT` (or `GOOGLE_CLOUD_PROJECT`) is not set. Set it once
in your shell profile:

```bash
export KINETIC_PROJECT="your-project-id"
```

Or pass `project=` to the decorator. See [Configuration](configuration.md).

### "404 Requested entity was not found"

A required GCP resource — usually an Artifact Registry repository or a
GKE cluster — doesn't exist yet. Run the setup once:

```bash
kinetic up
```

Or, if `up` already ran, enable the missing APIs and create the
registry manually (uncommon):

```bash
gcloud services enable compute.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com \
    container.googleapis.com --project=$KINETIC_PROJECT

gcloud artifacts repositories create "kn-${KINETIC_CLUSTER:-kinetic-cluster}" \
    --repository-format=docker --location=us \
    --project=$KINETIC_PROJECT
```

### Container build is slow on first run

The first run with a given `requirements.txt` builds a new container
image via Cloud Build (~2–5 minutes). Subsequent runs reuse the cached
image and start in under a minute. If you're churning dependencies
multiple times a day and this is hurting you, see
[Execution Modes](guides/execution_modes.md) for prebuilt mode.

### Every submit uploads for a long time

Kinetic logs the `context.zip` size on every submit. Kinetic logs a
warning above 100 MB, and lists the five largest files. Read that warning
first. The usual cause is a data directory, a checkpoint directory, or a
virtualenv inside the [package root](guides/packaging.md). Wrap large
inputs in `kinetic.Data(...)`, which Kinetic uploads one time and caches
by content hash. As an alternative, add the paths to a `.kineticignore`
file.

Kinetic logs a separate warning above 50 MB of *payload*. The payload
holds your arguments, the module-level globals that your function reads,
and your first-party module code. The payload holds no files from the
archive. Load a large object inside the function, or pass it as `Data`.

### Container build failures

Check Cloud Build logs:

```bash
gcloud builds list --project=$KINETIC_PROJECT --limit=5
gcloud builds log <build-id> --project=$KINETIC_PROJECT
```

Common causes: a package in `requirements.txt` that doesn't exist,
network issues during install, or a base image that's been updated
since you last built. See [Dependencies](guides/dependencies.md).

## Auth and config issues

### "Permission denied" on GCP operations

Your user (or service account) is missing IAM roles. The minimum set
for Kinetic is `roles/storage.admin`, `roles/artifactregistry.admin`,
`roles/container.admin`, and `roles/cloudbuild.builds.editor` on the
project:

```bash
gcloud projects add-iam-policy-binding $KINETIC_PROJECT \
    --member="user:your-email@example.com" \
    --role="roles/storage.admin"
```

Repeat for the other roles. `kinetic init`'s troubleshoot path flags
missing roles by checking the actual operations that fail.

### Application Default Credentials missing or expired

```bash
gcloud auth login
gcloud auth application-default login
```

If you've previously set `GOOGLE_APPLICATION_CREDENTIALS` to a service
account key, that takes precedence over user ADC.

### Settings aren't taking effect

Run `kinetic config` — it prints every config value and where it came
from (decorator arg, CLI flag, env var, or default). The precedence
rules are documented in [Configuration](configuration.md).

## Scheduling and quota issues

### Job stuck in `PENDING` for more than 10 minutes

The cluster autoscaler is trying to provision a node but can't. Two
common reasons:

- **No quota for the requested accelerator** in your zone. Check
  Cloud Console → IAM & Admin → Quotas, filter by your accelerator
  type. If quota is exhausted, request more or try a different zone.
- **Spot capacity is unavailable.** If your node pool was created with
  `--spot`, GCP may have no spot capacity to allocate right now.
  Switch to on-demand or try later.

`kinetic init`'s troubleshoot path includes a quota check that surfaces
exhausted accelerator quotas in your region. If it doesn't flag
anything, inspect the Cloud Console quota page directly for
finer-grained breakdowns.

### Multi-host TPU job fails right after submit

Likely causes: topology mismatch (your code expected a different number
of devices than the slice has), a stale Pathways context from a prior
crashed job, or one host failing before the others can join the
collective. See [Distributed Training](guides/distributed_training.md)
for the full list of multi-host failure modes.

## Runtime failures

### `ModuleNotFoundError` / `ImportError` on the remote pod

Three different causes produce the same exception. Identify your cause
from the module name in the traceback:

| The missing module is | The cause | The fix |
| --------------------- | --------- | ------- |
| A third-party package (`pandas`, `transformers`, and more) | The package is not in your `requirements.txt` or `pyproject.toml`. A local `pip install` does not carry over. | Add the package to the dependency file. Read the `Using dependency file: ...` log line. Kinetic walks up from the entry directory of your function, so Kinetic can select a different file than you expect. See [Dependencies](guides/dependencies.md). |
| One of your own modules or packages (`trainer`, `mylib.utils`, and more) | The module lives outside the detected [package root](guides/packaging.md), so it never entered `context.zip`. An exclusion rule can also remove it. | Put a `pyproject.toml`, a `requirements.txt`, or a `.git` at the top of the tree that you want to ship. As an alternative, set `KINETIC_PACKAGE_ROOT`. Check your `.kineticignore` and the default exclusion list. |
| `kinetic` itself | An old prebuilt image or custom image does not have Kinetic installed, and one of your own modules runs `import kinetic` at module scope. | Kinetic now ships your first-party modules by value, so the pod does not import them to unpickle the job. Rebuild your image: a bundled image installs `keras-kinetic` automatically. For a custom image, run `pip install keras-kinetic` in the image. |

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
2. **Did an exclusion rule remove it?** Kinetic skips `.venv`,
   `node_modules`, the cache directories, and every path that matches
   your `.kineticignore`. Kinetic also excludes a local path that you
   wrap in `Data(...)`. Read such a path through the value that the
   `Data` argument resolves to, and not through its original location.
3. **Is the path relative to the directory that you launched from?** The
   pod changes to the workspace directory that matches your client
   working directory, so a relative path behaves as it does locally. If
   your client working directory was *outside* the package root, the pod
   uses the workspace root instead.

An absolute client path is not a supported way to read a shipped file.
The runner does create one symbolic link at the path of your client
working directory. That link points at the workspace root. The link exists
so the debugger can map source files. Use a relative path.

### Pickle / cloudpickle errors at submit time

Your function, or one of its closures, references an object that
`cloudpickle` cannot serialize. These objects are the usual causes:

- An open file handle.
- A database connection.
- A thread.
- A lock.
- A module-level singleton built for local use.

Move the initialization of these objects inside the decorated function.

Kinetic bisects the payload after a failure, and names the component at
fault. One example message is
`kinetic could not serialize argument 2 (type socket): ...`. The message
identifies the function, one positional argument, or one keyword
argument, instead of a location inside `cloudpickle`. Kinetic counts
positional arguments from `0`.

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
(`client Python 3.12.2 / pod Python 3.11.9`). The runner reports a
difference in the `cloudpickle` version in the same way. The pod log holds
a skew warning only when the payload unpickled correctly.

Bundled mode always matches the Python of your client. For a prebuilt
image or a custom image, you must match it yourself. See
[Matching your local environment to the pod](guides/packaging.md#matching-your-local-environment-to-the-pod).

### JAX version mismatch errors

You probably pinned `jax` or `jaxlib` in `requirements.txt`. Kinetic
filters those out by default; if you need a specific version, use
`# kn:keep` (see [Dependencies](guides/dependencies.md)), but expect
to debug runtime/library alignment yourself.

### Job FAILS but logs look fine

The pod exited non-zero without writing a result payload — usually
caused by an OOM kill or the kernel reaping the process. Check pod
events with `kubectl describe pod <pod-name>` (find the pod name from
`kinetic jobs status <id>`).

### The job does not stop after your function returns

Your function left a non-daemon thread alive: a data-loader worker, a
metrics uploader, or a `ThreadPoolExecutor` that your code did not shut
down. CPython waits for non-daemon threads at interpreter exit. The pod
therefore stays alive after the work ends. You continue to pay for the
accelerator.

Kinetic uploads the result first. Kinetic then logs a warning that names
every thread still alive, and forces the process to exit. The job
completes, but that warning reports a real leak. Call `.shutdown()` or
`.join()` on your executors and pools. As an alternative, create your
threads with `daemon=True`.

### The job fails with "function called sys.exit(N)"

Kinetic treats a `sys.exit()` inside your function as the process-level
exit that it is. `sys.exit()` and `sys.exit(0)` report a **success**. The
result is `None`. Any other exit code fails the job with that message.
Return a value instead of an exit if you want Kinetic to collect a result.

## Missing outputs and results

### "Job failed but no result payload was found"

The pod died before it wrote anything to
`gs://{bucket}/{job_id}/result.pkl`. Kinetic uploads a failure payload
for each of its own startup phases. That payload carries a `phase` field,
which names the phase at fault:

- `artifact download`
- `artifact verification`
- `requirements install`
- `context extract`
- `payload unpickle`
- `environment setup`
- `data resolve`
- `debugger setup`

If you see this message, the failure happened outside those phases. These
causes remain:

- **Something killed the container** — an out-of-memory kill, a node
  preemption on Spot, or a reap by the kernel. Run
  `kubectl describe pod <pod-name>` and look for `OOMKilled` or an
  eviction event. Get the pod name from `kinetic jobs status <id>`.
- **The image cannot start the runner** — a custom image without
  `/app/remote_runner.py`, `python3`, `cloudpickle`,
  `google-cloud-storage`, or `absl-py`. The pod logs stop before any
  Kinetic output.
- **Cloud Storage was unreachable** — the runner could not upload the
  failure payload either. The pod logs hold the original error.
- **Kinetic already deleted the artifacts** — a blocking `run()` deletes
  them after it collects a result, and `run_async()` deletes them on
  `.cleanup()`. A reattach after that point finds nothing.

For a long job, write your artifacts under `KINETIC_OUTPUT_DIR` instead
of a return value. See [Checkpointing](guides/checkpointing.md).

### Kinetic cannot serialize the return value of your job

Your function returned an object that `cloudpickle` cannot serialize. The
runner then uploads a payload with the flag `serialization_failed` and a
`repr()` of the value. Kinetic reports the job as **failed**, because the
return value is not retrievable, and a success report would hide the
problem. The local error holds the truncated `repr`, so you can see what
your function returned.

Kinetic keeps the Cloud Storage artifacts in this case, so you can
inspect the run afterwards. Kinetic still deletes the Kubernetes
resource, so collect the pod logs first if you need them. Return
something serializable instead, such as a path or a dict of metrics.
Write the heavy object, or the object that holds a handle, to
`KINETIC_OUTPUT_DIR`.

### `result()` cannot unpickle the result after a reattach

The result references classes from your project, or from a package that
this client cannot import. Unpickling needs those types locally. A
reattach from a different directory, a different virtualenv, or a
different machine can therefore fail, although the client that submitted
the job succeeds.

Kinetic raises a `RuntimeError` that names the job and the artifact URI
(`gs://{bucket}/{job_id}/result.pkl`), and chains the original error.
Kinetic does **not** delete the Cloud Storage artifacts, so you lose
nothing. Reattach from the project directory, with the same packages
installed. As an alternative, download the object and inspect it there.

Return plain data to avoid the problem completely: a dict, a list, a
number, a string, or an array.

### Files I wrote inside the job are gone

Two possibilities:

- You wrote them under `/tmp` or another pod-local path. The pod is
  destroyed when the job ends; pod-local files don't survive. Always
  write to `KINETIC_OUTPUT_DIR`.
- You wrote them under `KINETIC_OUTPUT_DIR` but more than 30 days have
  passed. The default GCS bucket has a 30-day TTL. Copy critical
  artifacts to a bucket without lifecycle rules. See
  [Checkpointing](guides/checkpointing.md) for the TTL and retention
  details.

### Logs aren't streaming back

Network blip during a `--follow` stream is the most common cause. The
pod is unaffected — log retrieval is read-only. Use
`kinetic jobs logs <id>` (without `--follow`) or `--tail N` to fetch
fresh logs from any machine.

## What the troubleshoot path actually checks

`kinetic init`'s troubleshoot path runs eight groups of checks and
prints concrete fix commands when any fail. The groups (matching the
source at `kinetic/cli/commands/doctor.py`):

1. **Local Tools** — `gcloud`, `kubectl`, and
   `gke-gcloud-auth-plugin` are installed and on your PATH.
2. **Authentication** — Application Default Credentials are present,
   refreshable, and not expired.
3. **Configuration** — `KINETIC_PROJECT`, `KINETIC_ZONE`, and
   `KINETIC_CLUSTER` resolve to non-empty values.
4. **GCP Project** — the project exists and has billing enabled.
5. **GCP APIs** — Compute Engine, Cloud Build, Artifact Registry,
   Storage, and Container APIs are enabled.
6. **GCP Resources** — the Kinetic service accounts, Artifact Registry
   repository, GCS buckets, VPC network, and Cloud NAT all exist.
7. **Infrastructure** — Pulumi state is present and the GKE cluster is
   in the `RUNNING` state.
8. **Kubernetes** — your `kubeconfig` points at the cluster, the API
   server responds, node pools are healthy, GPU drivers are installed
   where needed, and accelerator quotas are not exhausted.

Each failing check prints a one-line fix suggestion. For multi-step
fixes, the troubleshoot path prints a copy-paste command block.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1em` Getting Started
:link: getting_started
:link-type: doc

First-run setup that shouldn't have to fail twice.
:::

:::{grid-item-card} {octicon}`file-directory;1em` What Ships to the Pod
:link: guides/packaging
:link-type: doc

The packaging contract behind most import failures and most failures
of a file path.
:::

:::{grid-item-card} {octicon}`book;1em` FAQ
:link: guides/faq
:link-type: doc

Quick answers to common conceptual confusions.
:::

:::{grid-item-card} {octicon}`gear;1em` Configuration
:link: configuration
:link-type: doc

Env vars and precedence.
:::
::::
