# Architecture Overview

Kinetic automates the process of running Python functions on Google Cloud
Platform (GCP) accelerators. It handles packaging, infrastructure provisioning,
and execution management to provide a seamless experience for remote workloads.

## Execution Lifecycle

When a function decorated with `@kinetic.run()` is executed (either directly
for a synchronous run, or via `run_async()` for a detached job), the system
follows these steps:

:::{container} kinetic-steps
1.  **Context Resolution**: Kinetic aggregates function parameters, environment variables, and local configurations into a unified `JobContext`.
2.  **Credential Validation**: The system verifies active `gcloud` and `kubectl` credentials, performing automatic configuration where necessary to ensure access to GCP services.
3.  **Artifact Preparation**:
    *   **Data Dependencies**: Local data paths are hashed and uploaded to Google Cloud Storage (GCS) if they are not already present in the content-addressed cache.
    *   **Function Serialization**: Kinetic serializes the decorated function and its closure with `cloudpickle`. Kinetic registers the modules under the package root for serialization by value. The remote pod thus does not import your first-party code to deserialize the job.
    *   **Project Packaging**: Kinetic compresses the package root into a ZIP archive. To find the package root, Kinetic first walks up out of any Python package. Kinetic then walks up to the nearest directory with a project marker. The markers are `pyproject.toml`, `requirements.txt`, `setup.py`, `setup.cfg`, and `.git`. `KINETIC_PACKAGE_ROOT` replaces this search. Kinetic excludes the paths that the Data API controls, the default exclusion list, and the `.kineticignore` patterns.
    *   **Packaging Plan**: Kinetic writes a packaging plan into the archive at `.kinetic/plan.json`. The plan holds the `sys.path` entries and the client working directory, both relative to the package root. The runner reads the plan to build its `sys.path` and to change its working directory. See [What Ships to the Pod](guides/packaging.md).
4.  **Container Image Management**: Kinetic generates a hash of project dependencies (e.g., `requirements.txt` or `pyproject.toml`). If a corresponding image does not exist in Artifact Registry, Kinetic initiates a Cloud Build job to create it.
5.  **Job Submission**: Based on the requested accelerator type, Kinetic submits a Kubernetes Job (for GKE) or a LeaderWorkerSet (for multi-host Pathways) to the target cluster.
6.  **Remote Execution**: The remote pod pulls the container image, retrieves the serialized artifacts, mounts the required data volumes, and executes the function.
7.  **Result Retrieval**: Upon completion, the function's return value is retrieved from GCS, deserialized, and returned to the local Python process.
:::
