# Architecture Overview

This page is for contributors. It maps the execution pipeline to the
modules in the `kinetic` package. For the user-level model, see
[How Kinetic Works](concepts.md).

## Modules

```text
kinetic/
├── core/           # @run decorator, RemoteCallable, accelerator registry and parser
├── backend/        # JobContext, GKE and Pathways backends, k8s helpers, log streaming
├── data/           # Data class, content hashing, data references
├── infra/          # Container image building and caching (Cloud Build)
├── runner/         # remote_runner.py — the entry point inside the pod
├── utils/          # Packager (payload, context.zip, packaging plan) and Cloud Storage helpers
├── jobs.py         # JobHandle, attach(), list_jobs()
├── collections.py  # run_async_map(), BatchHandle, attach_batch()
├── debug.py        # debugpy attach and port-forward helpers
├── cli/            # The `kinetic` command
│   ├── commands/   # init, up, down, status, config, pool, jobs, build-image, profile, accelerators
│   ├── infra/      # Pulumi program, stack and state management, post-deploy steps
│   ├── profiles.py # Profile store and the resolve_infra() precedence chain
│   └── options.py  # Shared --project/--zone/--cluster options
├── credentials.py  # gcloud, ADC, and kubeconfig checks and setup
└── constants.py    # Zone and region helpers, default names
```

## Execution lifecycle

A call to a decorated function, direct or through `run_async()`, goes
through these steps:

:::{container} kinetic-steps
1. **Context resolution.** `resolve_infra()` in `cli/profiles.py`
   resolves the project, zone, cluster, and namespace from the decorator
   arguments, the `KINETIC_*` environment variables, the active profile,
   and the defaults, in that order. `JobContext.from_params()` in
   `backend/execution.py` collects every other job setting into one
   mutable `JobContext`.
2. **Credential validation.** `credentials.py` verifies the `gcloud`
   login, Application Default Credentials, and the `kubeconfig` entry
   for the cluster, and configures them when it can.
3. **Artifact preparation.** `_prepare_artifacts()` in
   `backend/execution.py`:
   - Uploads each `Data` object to the content-addressed cache in the
     jobs bucket, and replaces the object with a data reference.
   - Registers each module under the package root for serialization by
     value, and serializes the function, its arguments, and the captured
     environment variables with `cloudpickle` into `payload.pkl`.
   - Resolves the package root: up out of every `__init__.py` package,
     then up to the nearest `pyproject.toml`, `requirements.txt`,
     `setup.py`, `setup.cfg`, or `.git`. `KINETIC_PACKAGE_ROOT` replaces
     the search. Archives the root into `context.zip`, without the
     `Data` paths, the default exclusions, and the `.kineticignore`
     matches.
   - Writes a packaging plan into the archive at `.kinetic/plan.json`.
     The plan holds the client `sys.path` entries and the client working
     directory, relative to the package root. See
     [What Ships to the Pod](guides/packaging.md).
4. **Container image and upload, in parallel.**
   `infra/container_builder.py` hashes the base image, the accelerator
   category, the Kinetic version, the filtered dependency file, the
   runner script, and the Dockerfile template. If Artifact Registry has
   no image with that tag, it runs Cloud Build. Prebuilt mode resolves a
   base image instead; custom image mode uses the URI as given.
5. At the same time, `utils/storage.py` uploads `payload.pkl`,
   `context.zip`, and, in prebuilt mode, `requirements.txt` to
   `gs://{jobs bucket}/{job_id}/`. The client records the SHA-256 hash of
   the payload and of the archive.
6. **Submission.** `GKEBackend` creates a Kubernetes Job for a
   single-host accelerator. `PathwaysBackend` creates a LeaderWorkerSet
   for a multi-host TPU slice. `kinetic.run()` selects the backend from
   `TpuConfig.num_nodes` unless `backend=` is set. Both backends pass
   the artifact hashes in the pod specification.
7. **Remote execution.** `runner/remote_runner.py` downloads both
   artifacts, verifies the hashes, extracts the archive, rebuilds
   `sys.path` and the working directory from the plan, installs the
   dependency file in prebuilt mode, resolves the `Data` references and
   volumes, applies the captured environment variables, sets
   `KINETIC_OUTPUT_DIR`, and calls the function.
8. **Result retrieval.** The runner writes a result payload to
   `gs://{jobs bucket}/{job_id}/result.pkl`. `JobHandle.result()`
   downloads it, returns the value or raises the exception with the
   remote traceback, and, by default, deletes the Kubernetes resource
   and, when the job succeeded, the job artifacts.
:::

## Result payload

```python
{
  "success": bool,
  "result": Any,  # if success is True
  "exception": Exception,  # if success is False
  "traceback": str,  # if success is False
}
```

When one of its own startup phases fails, the runner writes a failure
payload with a `phase` field. When the return value cannot be pickled,
the runner writes a payload with the flag `serialization_failed`. See
[Troubleshooting](troubleshooting.md).

## Backend selection

- CPU, GPU, and single-host TPU → `GKEBackend` (Kubernetes Job).
- Multi-host TPU (`TpuConfig.num_nodes > 1`, one node per host) →
  `PathwaysBackend` (LeaderWorkerSet).
- An explicit `backend=` argument overrides the selection.

## Infrastructure state

The CLI keeps three layers of state: the in-memory `InfraConfig`, the
Pulumi stack in the state bucket `gs://{project}-kinetic-state`, and the
Google Cloud resources. Each `(project, cluster)` pair has its own stack,
named `{project}-{cluster}`. All stack operations go through
`cli/infra/state.py`: `load_state()`, `apply_update()`, and
`apply_destroy()`. See `AGENTS.md` in the repository for the conventions
that contributors follow.
