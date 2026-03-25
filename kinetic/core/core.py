import functools
import os

from kinetic.backend.execution import (
  GKEBackend,
  JobContext,
  PathwaysBackend,
  submit_remote,
)
from kinetic.constants import DEFAULT_CLUSTER_NAME
from kinetic.core import accelerators
from kinetic.data import Data


def _validate_volumes(volumes):
  """Validate the optional volumes mapping."""
  if volumes is None:
    return
  if not isinstance(volumes, dict):
    raise TypeError(f"volumes must be a dict, got {type(volumes).__name__}")
  for mount_path, data_obj in volumes.items():
    if not isinstance(mount_path, str) or not mount_path.startswith("/"):
      raise ValueError(
        f"Volume mount path must be an absolute path "
        f"(start with '/'), got: {mount_path!r}"
      )
    if not isinstance(data_obj, Data):
      raise TypeError(
        f"Volume value for {mount_path!r} must be a Data "
        f"instance, got {type(data_obj).__name__}"
      )


def _capture_env(capture_env_vars):
  """Capture requested environment variables for remote execution."""
  env_vars = {}
  if not capture_env_vars:
    return env_vars

  for pattern in capture_env_vars:
    if pattern.endswith("*"):
      prefix = pattern[:-1]
      env_vars.update(
        {k: v for k, v in os.environ.items() if k.startswith(prefix)}
      )
    elif pattern in os.environ:
      env_vars[pattern] = os.environ[pattern]
  return env_vars


def _resolve_backend_name(accelerator, backend):
  """Resolve the backend from explicit config or accelerator type."""
  if backend is not None:
    return backend

  try:
    accel_config = accelerators.parse_accelerator(accelerator)
    if (
      isinstance(accel_config, accelerators.TpuConfig)
      and accel_config.num_nodes > 1
    ):
      return "pathways"
  except ValueError:
    pass
  return "gke"


def _build_context(
  func,
  args,
  kwargs,
  accelerator,
  container_image,
  zone,
  project,
  cluster,
  namespace,
  env_vars,
  volumes,
  resolved_backend,
):
  """Create a (JobContext, BaseK8sBackend) pair with resolved defaults."""
  if not cluster:
    cluster = os.environ.get("KINETIC_CLUSTER", DEFAULT_CLUSTER_NAME)
  if not namespace:
    namespace = os.environ.get("KINETIC_NAMESPACE", "default")

  ctx = JobContext.from_params(
    func,
    args,
    kwargs,
    accelerator,
    container_image,
    zone,
    project,
    env_vars,
    cluster_name=cluster,
    volumes=volumes,
  )

  if resolved_backend == "pathways":
    backend_inst = PathwaysBackend(cluster=cluster, namespace=namespace)
  else:
    backend_inst = GKEBackend(cluster=cluster, namespace=namespace)

  return ctx, backend_inst


def run(
  accelerator="v6e-8",
  container_image=None,
  zone=None,
  project=None,
  capture_env_vars=None,
  cluster=None,
  backend=None,
  namespace=None,
  volumes=None,
):
  """Execute function on remote TPU/GPU.

  Args:
    accelerator: TPU/GPU type (e.g., 'v3-8', 'v5litepod-4', 'l4', 'a100')
    container_image: Custom container image URI (optional)
    zone: GCP zone (default: from KINETIC_ZONE or 'us-central1-a')
    project: GCP project (default: from KINETIC_PROJECT)
    capture_env_vars: List of environment variable names or patterns (ending in *)
      to propagate to the remote environment. Defaults to None.
    cluster: GKE cluster name (default: from KINETIC_CLUSTER)
    backend: Backend to use ('gke' or 'pathways')
    namespace: Kubernetes namespace (default: None, resolved via
      KINETIC_NAMESPACE env var or 'default')
    volumes: Dict mapping absolute mount paths to Data objects, e.g.
      ``{"/data": Data("./dataset/")}``. Data is downloaded to these
      paths on the pod before function execution.
  """
  _validate_volumes(volumes)

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      env_vars = _capture_env(capture_env_vars)
      resolved_backend = _resolve_backend_name(accelerator, backend)

      if resolved_backend not in ("gke", "pathways"):
        raise ValueError(
          f"Unknown backend: {resolved_backend}. "
          "Use 'gke', 'pathways', or None for auto-detection"
        )

      return _execute_on_backend(
        func,
        args,
        kwargs,
        accelerator,
        container_image,
        zone,
        project,
        cluster,
        namespace,
        env_vars,
        volumes,
        resolved_backend,
      )

    return wrapper

  return decorator


def submit(
  accelerator="v6e-8",
  container_image=None,
  zone=None,
  project=None,
  capture_env_vars=None,
  cluster=None,
  backend=None,
  namespace=None,
  volumes=None,
):
  """Submit function for remote execution, returning a ``JobHandle``.

  Same parameters as ``run()``.  Blocks through container build and
  artifact upload, but returns immediately after k8s submission.
  Use the returned ``JobHandle`` to observe, collect, or cancel.

  Returns:
    A decorator whose wrapper returns a ``JobHandle``.
  """
  _validate_volumes(volumes)

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      env_vars = _capture_env(capture_env_vars)
      resolved_backend = _resolve_backend_name(accelerator, backend)

      if resolved_backend not in ("gke", "pathways"):
        raise ValueError(
          f"Unknown backend: {resolved_backend}. "
          "Use 'gke', 'pathways', or None for auto-detection"
        )

      return _submit_on_backend(
        func,
        args,
        kwargs,
        accelerator,
        container_image,
        zone,
        project,
        cluster,
        namespace,
        env_vars,
        volumes,
        resolved_backend,
      )

    return wrapper

  return decorator


# ------------------------------------------------------------------
# Internal dispatch helpers
# ------------------------------------------------------------------


def _execute_on_backend(
  func,
  args,
  kwargs,
  accelerator,
  container_image,
  zone,
  project,
  cluster,
  namespace,
  env_vars,
  volumes,
  resolved_backend,
):
  """Build context and execute synchronously (submit + result)."""
  return _submit_on_backend(
    func,
    args,
    kwargs,
    accelerator,
    container_image,
    zone,
    project,
    cluster,
    namespace,
    env_vars,
    volumes,
    resolved_backend,
  ).result()


def _submit_on_backend(
  func,
  args,
  kwargs,
  accelerator,
  container_image,
  zone,
  project,
  cluster,
  namespace,
  env_vars,
  volumes,
  resolved_backend,
):
  """Build context and submit asynchronously."""
  ctx, backend_inst = _build_context(
    func,
    args,
    kwargs,
    accelerator,
    container_image,
    zone,
    project,
    cluster,
    namespace,
    env_vars,
    volumes,
    resolved_backend,
  )
  return submit_remote(ctx, backend_inst)
