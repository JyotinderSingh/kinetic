"""Unified remote execution module for GKE backend.

This module consolidates the common execution logic shared between different
backend implementations, reducing code duplication and improving maintainability.
"""

import abc
import concurrent.futures
import hashlib
import inspect
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from absl import logging

from kinetic.backend import gke_client, k8s_utils, pathways_client
from kinetic.constants import (
  build_bucket_name,
  get_default_cluster_name,
  get_default_zone,
  get_required_project,
  zone_to_region,
)
from kinetic.credentials import ensure_credentials
from kinetic.data import make_data_ref
from kinetic.infra import container_builder
from kinetic.jobs import JobHandle
from kinetic.utils import packager, storage


@dataclass
class JobContext:
  """Encapsulates all state for a remote job execution."""

  # Function and arguments
  func: Callable
  args: tuple
  kwargs: dict
  env_vars: dict

  # Configuration
  accelerator: str
  container_image: Optional[str]
  zone: str
  project: str
  cluster_name: str
  base_image_repo: Optional[str] = None
  working_dir: Optional[str] = None

  # Generated identifiers
  job_id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:8]}")

  # Derived values (computed in __post_init__)
  bucket_name: str = field(init=False)
  region: str = field(init=False)
  display_name: str = field(init=False)

  # Data volumes {mount_path: Data}
  volumes: Optional[dict] = None

  # FUSE volume specs for pod spec generation (not serialized into payload)
  fuse_volume_specs: Optional[list[dict]] = None

  # Configuration modifiers
  spot: bool = False
  debug: bool = False
  output_dir: Optional[str] = None

  # Artifact paths (set during prepare phase)
  payload_path: Optional[str] = None
  context_path: Optional[str] = None
  requirements_path: Optional[str] = None  # requirements.txt or pyproject.toml
  image_uri: Optional[str] = None
  payload_sha256: Optional[str] = None
  context_sha256: Optional[str] = None

  def __post_init__(self):
    self.bucket_name = build_bucket_name(self.project, self.cluster_name)
    self.region = zone_to_region(self.zone)
    func_name = getattr(self.func, "__name__", None) or type(self.func).__name__
    self.display_name = f"kinetic-{func_name}-{self.job_id}"
    if self.working_dir is None:
      self.working_dir = _resolve_working_dir(self.func)

    if not self.output_dir:
      self.output_dir = f"gs://{self.bucket_name}/outputs/{self.job_id}"
    self.env_vars["KINETIC_OUTPUT_DIR"] = self.output_dir

  @classmethod
  def from_params(
    cls,
    func: Callable,
    args: tuple,
    kwargs: dict,
    accelerator: str,
    container_image: Optional[str],
    zone: Optional[str],
    project: Optional[str],
    env_vars: dict,
    cluster_name: Optional[str] = None,
    volumes: Optional[dict] = None,
    spot: bool = False,
    debug: bool = False,
    output_dir: Optional[str] = None,
    base_image_repo: Optional[str] = None,
  ) -> "JobContext":
    """Factory method with default resolution for zone/project/cluster."""
    if not zone:
      zone = get_default_zone()
    if not project:
      project = get_required_project()
    if not cluster_name:
      cluster_name = get_default_cluster_name()

    return cls(
      func=func,
      args=args,
      kwargs=kwargs,
      env_vars=env_vars,
      accelerator=accelerator,
      container_image=container_image,
      base_image_repo=base_image_repo,
      zone=zone,
      project=project,
      cluster_name=cluster_name,
      working_dir=_resolve_working_dir(func),
      volumes=volumes,
      spot=spot,
      debug=debug,
      output_dir=output_dir or os.environ.get("KINETIC_OUTPUT_DIR"),
    )


class BaseK8sBackend(abc.ABC):
  """Base class for Kubernetes-based backends."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """The unique backend identifier, e.g., 'gke' or 'pathways'."""

  def __init__(self, cluster: str, namespace: str = "default"):
    self.cluster = cluster
    self.namespace = namespace

  def validate_preflight(self, ctx: JobContext) -> None:  # noqa: B027
    """Perform preflight checks before building container or uploading artifacts."""

  @abc.abstractmethod
  def submit_job(self, ctx: JobContext) -> Any:
    """Submit a job to the backend. Returns backend-specific job handle."""

  @abc.abstractmethod
  def wait_for_job(self, job: Any, ctx: JobContext) -> None:
    """Wait for job completion. Raises RuntimeError if job fails."""

  @abc.abstractmethod
  def cleanup_job(
    self,
    job: Any,
    ctx: JobContext,
    timeout: float = 180,
    poll_interval: float = 2,
  ) -> None:
    """Clean up backend resources after job completion."""

  @abc.abstractmethod
  def get_k8s_name(self, job_id: str) -> str:
    """Return the backend-specific Kubernetes resource name."""

  @abc.abstractmethod
  def job_exists(self, job_name: str) -> bool:
    """Return whether the Kubernetes resource currently exists."""


class GKEBackend(BaseK8sBackend):
  """Backend adapter for standard GKE Jobs."""

  name = "gke"

  def validate_preflight(self, ctx: JobContext) -> None:
    """Check if the required node pool exists for the accelerator."""
    k8s_utils.validate_preflight(accelerator=ctx.accelerator)

  def submit_job(self, ctx: JobContext) -> Any:
    """Submit job to GKE cluster."""
    logging.info("Submitting job to GKEBackend...")
    requirements_uri = _requirements_uri(ctx)
    return gke_client.submit_k8s_job(
      display_name=ctx.display_name,
      container_uri=ctx.image_uri,
      accelerator=ctx.accelerator,
      project=ctx.project,
      job_id=ctx.job_id,
      bucket_name=ctx.bucket_name,
      namespace=self.namespace,
      spot=ctx.spot,
      requirements_uri=requirements_uri,
      fuse_volume_specs=ctx.fuse_volume_specs,
      debug=ctx.debug,
      payload_sha256=ctx.payload_sha256,
      context_sha256=ctx.context_sha256,
    )

  def wait_for_job(self, job: Any, ctx: JobContext) -> None:
    """Wait for GKE job completion."""
    gke_client.wait_for_job(job, namespace=self.namespace)

  def cleanup_job(
    self,
    job: Any,
    ctx: JobContext,
    timeout: float = 180,
    poll_interval: float = 2,
  ) -> None:
    """Clean up K8s job resources."""
    job_name = job.metadata.name
    gke_client.cleanup_job(
      job_name,
      namespace=self.namespace,
      timeout=timeout,
      poll_interval=poll_interval,
    )

  def get_k8s_name(self, job_id: str) -> str:
    """Return the standard GKE Job name for this job ID."""
    return f"kinetic-{job_id}"

  def job_exists(self, job_name: str) -> bool:
    """Return whether the GKE Job exists."""
    return gke_client.job_exists(job_name, namespace=self.namespace)


class PathwaysBackend(BaseK8sBackend):
  """Backend adapter for ML Pathways using LeaderWorkerSet."""

  name = "pathways"

  def validate_preflight(self, ctx: JobContext) -> None:
    """Preflight checks for Pathways (currently same as GKE)."""
    k8s_utils.validate_preflight(accelerator=ctx.accelerator)

  def submit_job(self, ctx: JobContext) -> Any:
    """Submit LWS job to GKE cluster."""
    logging.info("Submitting job to PathwaysBackend...")
    requirements_uri = _requirements_uri(ctx)
    return pathways_client.submit_pathways_job(
      display_name=ctx.display_name,
      container_uri=ctx.image_uri,
      accelerator=ctx.accelerator,
      project=ctx.project,
      job_id=ctx.job_id,
      bucket_name=ctx.bucket_name,
      namespace=self.namespace,
      spot=ctx.spot,
      requirements_uri=requirements_uri,
      fuse_volume_specs=ctx.fuse_volume_specs,
      debug=ctx.debug,
      payload_sha256=ctx.payload_sha256,
      context_sha256=ctx.context_sha256,
    )

  def wait_for_job(self, job: Any, ctx: JobContext) -> None:
    """Wait for Pathways LWS completion."""
    pathways_client.wait_for_job(ctx.job_id, namespace=self.namespace)

  def cleanup_job(
    self,
    job: Any,
    ctx: JobContext,
    timeout: float = 180,
    poll_interval: float = 2,
  ) -> None:
    """Clean up LWS resources."""
    job_name = pathways_client._get_job_name(ctx.job_id)
    pathways_client.cleanup_job(
      job_name,
      namespace=self.namespace,
      timeout=timeout,
      poll_interval=poll_interval,
    )

  def get_k8s_name(self, job_id: str) -> str:
    """Return the standard LeaderWorkerSet name for this job ID."""
    return pathways_client._get_job_name(job_id)

  def job_exists(self, job_name: str) -> bool:
    """Return whether the LeaderWorkerSet exists."""
    return pathways_client.job_exists(job_name, namespace=self.namespace)


# Files that mark the root of a project/repository for packaging purposes.
_PROJECT_MARKERS = (
  "pyproject.toml",
  "requirements.txt",
  "setup.py",
  "setup.cfg",
  ".git",
)


def _home_dir() -> str:
  """Return the absolute, normalized home directory."""
  return os.path.normpath(os.path.abspath(os.path.expanduser("~")))


def _has_repo_marker(search_dir: str) -> bool:
  """Return whether *search_dir* is a repository root.

  `.git` is a directory in normal clones and a file in worktrees and
  submodules, so mere existence is the test.
  """
  return os.path.exists(os.path.join(search_dir, ".git"))


def _has_project_marker(search_dir: str) -> bool:
  """Return whether *search_dir* looks like a project/repository root."""
  return any(
    os.path.exists(os.path.join(search_dir, marker))
    for marker in _PROJECT_MARKERS
  )


def _select_dependency_file(search_dir: str) -> Optional[str]:
  """Return the dependency file to use from a single directory level."""
  req_path = os.path.join(search_dir, "requirements.txt")
  pyproject_path = os.path.join(search_dir, "pyproject.toml")
  has_req = os.path.isfile(req_path)
  has_pyproject = os.path.isfile(pyproject_path)
  if has_req and has_pyproject:
    logging.info(
      "Both requirements.txt and pyproject.toml exist in %s; using "
      "requirements.txt (it wins at the same directory level)",
      search_dir,
    )
  if has_req:
    return req_path
  if has_pyproject:
    return pyproject_path
  return None


def _warn_if_no_dependencies(path: str) -> None:
  """Warn when the selected pyproject.toml declares no dependencies."""
  if not path.endswith("pyproject.toml"):
    return
  try:
    content = container_builder.prepare_requirements_content(path)
  except Exception:
    # Parsing errors surface with full context later in the build phase.
    return
  if content:
    return
  logging.warning(
    "pyproject.toml at %s declares no [project].dependencies; the job will "
    "run with base-image packages only. If your dependencies live in a "
    "requirements.txt at a parent directory, move or copy it next to your "
    "code (requirements.txt wins at the same level).",
    path,
  )


def _find_requirements(start_dir: str) -> Optional[str]:
  """Search up directory tree for requirements.txt or pyproject.toml.

  At each directory level, `requirements.txt` is preferred over
  `pyproject.toml`.  The first match found while walking towards the
  filesystem root is returned.  The walk stops after examining the first
  directory that contains a repository marker (`.git`, as a directory or
  as a worktree/submodule file), at the user's home directory, and at the
  filesystem root, so a stray dependency file above the project is never
  adopted as the job's entire dependency set.
  """
  start = os.path.normpath(os.path.abspath(start_dir))
  home = _home_dir()
  search_dir = start
  selected = None
  repo_root = None
  while True:
    if selected is None:
      selected = _select_dependency_file(search_dir)
    if _has_repo_marker(search_dir):
      repo_root = search_dir
      break
    parent_dir = os.path.dirname(search_dir)
    if search_dir == home or parent_dir == search_dir:
      break
    search_dir = parent_dir

  if selected is None:
    return None

  logging.info("Using dependency file: %s", selected)
  if repo_root is None and os.path.dirname(selected) != start:
    logging.warning(
      "Dependency file %s lives outside your code directory %s and no "
      "repository marker (.git) was found in between; its packages become "
      "the job's entire dependency set.",
      selected,
      start,
    )
  _warn_if_no_dependencies(selected)
  return selected


def _relpath_under(path: str, root: str) -> Optional[str]:
  """Return the relative path of *path* under *root*, or None if outside.

  Returns "" when the two denote the same directory.  Comparison retries
  on realpaths so symlinked roots (``/tmp`` vs ``/private/tmp``) match.
  """
  candidates = (
    (os.path.normpath(path), os.path.normpath(root)),
    (os.path.realpath(path), os.path.realpath(root)),
  )
  for candidate, base in candidates:
    if candidate == base:
      return ""
    if candidate.startswith(base.rstrip(os.sep) + os.sep):
      return os.path.relpath(candidate, base)
  return None


def _maybe_exclude(data_path, zip_root, exclude_paths) -> None:
  """Add data_path to exclude_paths if it's inside the packaged tree.

  The path is re-anchored on *zip_root* so it matches what the archiver
  walks even when the caller resolved symlinks differently.
  """
  data_abs = os.path.normpath(os.path.abspath(data_path))
  if not os.path.exists(data_abs):
    return
  root_abs = os.path.normpath(os.path.abspath(zip_root))
  rel = _relpath_under(data_abs, root_abs)
  if rel is None:
    return
  exclude_paths.add(
    os.path.normpath(os.path.join(root_abs, rel)) if rel else root_abs
  )


def _resolve_working_dir(func: Callable) -> str:
  """Resolve the user working directory from the wrapped function."""
  module = inspect.getmodule(func)
  module_file = getattr(module, "__file__", None) if module else None
  if module_file:
    return os.path.dirname(os.path.abspath(module_file))
  cwd = os.getcwd()
  logging.info(
    "Function %s has no source file (notebook, REPL, or 'python -c'); "
    "packaging the current directory instead: %s",
    getattr(func, "__name__", repr(func)),
    cwd,
  )
  return cwd


@dataclass
class PackagingPlan:
  """What to package for a job and how to rebuild the layout remotely.

  Attributes:
      working_dir: Directory of the file that defines the job function.
      package_root: Directory zipped into ``context.zip``.
      entry_rel: Relative path from *package_root* to *working_dir* ("" when
          they are the same directory).
      client_cwd_rel: Relative path of the client's cwd under
          *package_root*, or None when the cwd is outside the tree.
      sys_path_rel: Relative paths of client ``sys.path`` entries that live
          under *package_root*; always starts with "" (the root itself).
  """

  working_dir: str
  package_root: str
  entry_rel: str
  client_cwd_rel: Optional[str]
  sys_path_rel: list[str]


def _escape_package(working_dir: str, home: str) -> str:
  """Walk up out of any package the working dir belongs to."""
  search_dir = working_dir
  while os.path.exists(os.path.join(search_dir, "__init__.py")):
    parent_dir = os.path.dirname(search_dir)
    if parent_dir == search_dir or search_dir == home:
      break
    search_dir = parent_dir
  return search_dir


def _find_package_root(package_base: str, home: str) -> Optional[str]:
  """Search up from *package_base* for a project marker directory.

  `$HOME` and the filesystem root are never adopted as the packaging root
  unless the search started there.
  """
  search_dir = package_base
  while True:
    bounded = search_dir == home or os.path.dirname(search_dir) == search_dir
    if (not bounded or search_dir == package_base) and _has_project_marker(
      search_dir
    ):
      return search_dir
    if bounded:
      return None
    search_dir = os.path.dirname(search_dir)


def _sys_path_rel(package_root: str) -> list[str]:
  """Return relative client sys.path entries living under *package_root*."""
  rels = set()
  for entry in sys.path:
    # sys.path may contain non-string entries added by importer hooks.
    if not isinstance(entry, str) or not entry or not os.path.exists(entry):
      continue
    if "site-packages" in entry or "dist-packages" in entry:
      continue
    rel = _relpath_under(os.path.abspath(entry), package_root)
    if rel:
      rels.add(rel)
  return [""] + sorted(rels)


def compute_packaging_plan(
  func: Callable, working_dir: Optional[str] = None
) -> PackagingPlan:
  """Compute what to package for *func* and how the pod should lay it out.

  The packaging root is the directory that keeps the function's module
  importable under its original name: the walk escapes any package the
  defining file belongs to (directories holding `__init__.py`), then
  continues up to the nearest project marker (`pyproject.toml`,
  `requirements.txt`, `setup.py`, `setup.cfg`, `.git`).  `$HOME` and the
  filesystem root are bounds, never roots.  `KINETIC_PACKAGE_ROOT`
  overrides detection entirely.

  Args:
      func: The user function being submitted.
      working_dir: Overrides the resolved directory of the defining file.

  Returns:
      The `PackagingPlan` for this submission.

  Raises:
      ValueError: If `KINETIC_PACKAGE_ROOT` is not a parent of (or equal
          to) the directory being packaged.
  """
  resolved = working_dir or _resolve_working_dir(func)
  resolved = os.path.normpath(os.path.abspath(resolved))
  home = _home_dir()

  override = os.environ.get("KINETIC_PACKAGE_ROOT")
  if override:
    package_root = os.path.normpath(
      os.path.abspath(os.path.expanduser(override))
    )
    if not os.path.isdir(package_root):
      raise ValueError(
        f"KINETIC_PACKAGE_ROOT={override!r} (resolved to {package_root}) "
        f"is not an existing directory."
      )
    if _relpath_under(resolved, package_root) is None:
      raise ValueError(
        f"KINETIC_PACKAGE_ROOT={override!r} (resolved to {package_root}) "
        f"does not contain the code being packaged ({resolved}). Set it to "
        f"a parent directory of your project, or unset it."
      )
  else:
    package_base = _escape_package(resolved, home)
    package_root = _find_package_root(package_base, home) or package_base

  entry_rel = _relpath_under(resolved, package_root) or ""
  plan = PackagingPlan(
    working_dir=resolved,
    package_root=package_root,
    entry_rel=entry_rel,
    client_cwd_rel=_relpath_under(os.getcwd(), package_root),
    sys_path_rel=_sys_path_rel(package_root),
  )
  logging.info(
    "Packaging root: %s (code directory: %s)",
    plan.package_root,
    plan.entry_rel or ".",
  )
  return plan


_FUSE_DATA_MOUNT_PREFIX = "/_kinetic/fuse-data"


def _fuse_gcs_uri(gcs_uri: str, data_obj) -> str:
  """Return a file-level GCS URI for FUSE single-file mounts.

  For uploaded local single files, upload_data returns a directory-level
  URI (the hash prefix).  Append the filename so build_gcs_fuse_volumes
  scopes only-dir to the hash directory, not the entire data-cache/ tree.
  GCS-native URIs and directories are returned unchanged.
  """
  if not data_obj.is_dir and not data_obj.is_gcs:
    return f"{gcs_uri}/{os.path.basename(data_obj.path)}"
  return gcs_uri


def _process_volumes(
  ctx: JobContext, zip_root: str, exclude_paths: set[str]
) -> tuple[list[dict], list[dict]]:
  """Upload volume Data objects and build refs + FUSE specs.

  Args:
      ctx: Job context containing volume definitions, bucket name, and project.
      zip_root: Absolute path of the directory archived into context.zip,
          used to decide which Data paths to exclude from it.
      exclude_paths: Mutable set of local paths to exclude from the working
          directory archive. Paths of non-GCS Data objects are added here so
          they are not redundantly packaged.

  Returns:
      Tuple of (volume_refs, fuse_specs) where *volume_refs* are serializable
      data-ref dicts to embed in the payload, and *fuse_specs* are GCS FUSE
      mount descriptors for volumes that requested lazy loading.
  """
  volume_refs: list[dict] = []
  fuse_specs: list[dict] = []
  if not ctx.volumes:
    return volume_refs, fuse_specs

  for mount_path, data_obj in ctx.volumes.items():
    if mount_path.startswith(_FUSE_DATA_MOUNT_PREFIX + "/"):
      raise ValueError(
        f"Volume mount path {mount_path!r} is reserved for "
        f"auto-generated FUSE mounts. Use a different path."
      )
    gcs_uri = storage.upload_data(ctx.bucket_name, data_obj, ctx.project)
    volume_refs.append(
      make_data_ref(
        gcs_uri,
        data_obj.is_dir,
        mount_path=mount_path,
        fuse=data_obj.fuse,
        hf_trust_remote_code=data_obj.hf_trust_remote_code,
      )
    )
    if data_obj.fuse:
      fuse_specs.append(
        {
          "gcs_uri": _fuse_gcs_uri(gcs_uri, data_obj),
          "mount_path": mount_path,
          "is_dir": data_obj.is_dir,
          "read_only": True,
        }
      )
    if not data_obj.is_gcs:
      _maybe_exclude(data_obj.path, zip_root, exclude_paths)

  return volume_refs, fuse_specs


def _process_data_args(
  ctx: JobContext, zip_root: str, exclude_paths: set[str]
) -> tuple[dict[int, dict], list[dict]]:
  """Upload Data objects found in function args and build ref map + FUSE specs.

  Args:
      ctx: Job context containing the function args/kwargs, bucket name, and
          project.
      zip_root: Absolute path of the directory archived into context.zip,
          used to decide which Data paths to exclude from it.
      exclude_paths: Mutable set of local paths to exclude from the working
          directory archive. Paths of non-GCS Data objects are added here so
          they are not redundantly packaged.

  Returns:
      Tuple of (ref_map, fuse_specs) where *ref_map* is a dict keyed by
      ``id(data_obj)`` mapping to serializable data-ref dicts, and
      *fuse_specs* are GCS FUSE mount descriptors for args that requested
      lazy loading.
  """
  ref_map = {}
  fuse_specs = []
  fuse_counter = 0

  for data_obj, _ in packager.extract_data_refs(ctx.args, ctx.kwargs):
    gcs_uri = storage.upload_data(ctx.bucket_name, data_obj, ctx.project)
    if data_obj.fuse:
      mount_path = f"{_FUSE_DATA_MOUNT_PREFIX}/{fuse_counter}"
      fuse_counter += 1
      fuse_specs.append(
        {
          "gcs_uri": _fuse_gcs_uri(gcs_uri, data_obj),
          "mount_path": mount_path,
          "is_dir": data_obj.is_dir,
          "read_only": True,
        }
      )
      ref_map[id(data_obj)] = make_data_ref(
        gcs_uri,
        data_obj.is_dir,
        mount_path=mount_path,
        fuse=True,
        hf_trust_remote_code=data_obj.hf_trust_remote_code,
      )
    else:
      ref_map[id(data_obj)] = make_data_ref(
        gcs_uri,
        data_obj.is_dir,
        hf_trust_remote_code=data_obj.hf_trust_remote_code,
      )
    if not data_obj.is_gcs:
      _maybe_exclude(data_obj.path, zip_root, exclude_paths)

  return ref_map, fuse_specs


def _file_sha256(path: str) -> str:
  """Compute SHA-256 hash of a file."""
  hasher = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def _prepare_artifacts(ctx: JobContext, tmpdir: str) -> None:
  """Package function payload and working directory context."""
  logging.info("Packaging function and context...")
  if ctx.working_dir is None:
    raise ValueError("working_dir must be set before prepare")
  plan = compute_packaging_plan(ctx.func, working_dir=ctx.working_dir)
  zip_root = plan.package_root
  plan_fields = {
    "package_root": plan.package_root,
    "entry_rel": plan.entry_rel,
    "client_cwd_rel": plan.client_cwd_rel,
    "sys_path_rel": plan.sys_path_rel,
  }
  exclude_paths: set[str] = set()

  # Upload Data objects and build serializable refs
  volume_refs, vol_fuse = _process_volumes(ctx, zip_root, exclude_paths)
  ref_map, arg_fuse = _process_data_args(ctx, zip_root, exclude_paths)

  all_fuse = vol_fuse + arg_fuse
  ctx.fuse_volume_specs = all_fuse if all_fuse else None

  # Replace Data with refs in args/kwargs
  if ref_map:
    ctx.args, ctx.kwargs = packager.replace_data_with_refs(
      ctx.args, ctx.kwargs, ref_map
    )

  # Serialize function + args (with volume refs)
  ctx.payload_path = os.path.join(tmpdir, "payload.pkl")
  packager.save_payload(
    ctx.func,
    ctx.args,
    ctx.kwargs,
    ctx.env_vars,
    ctx.payload_path,
    volumes=volume_refs or None,
    working_dir=ctx.working_dir,
    package_root=plan.package_root,
    payload_extra=plan_fields,
  )
  ctx.payload_sha256 = _file_sha256(ctx.payload_path)
  logging.info("Payload serialized to %s", ctx.payload_path)

  # Zip the packaging root (excluding Data paths)
  ctx.context_path = os.path.join(tmpdir, "context.zip")
  packager.zip_working_dir(
    zip_root,
    ctx.context_path,
    exclude_paths=exclude_paths,
    plan_json=plan_fields,
  )
  ctx.context_sha256 = _file_sha256(ctx.context_path)
  logging.info("Context packaged to %s", ctx.context_path)

  # Find requirements.txt or pyproject.toml
  ctx.requirements_path = _find_requirements(ctx.working_dir)
  if not ctx.requirements_path:
    logging.info("No requirements.txt or pyproject.toml found")


def _is_prebuilt(ctx: JobContext) -> bool:
  """Return True if prebuilt image mode is active."""
  return ctx.container_image == "prebuilt"


def _build_container(ctx: JobContext) -> str:
  """Build or get cached container image. Returns the image URI."""
  if _is_prebuilt(ctx):
    image_uri = container_builder.get_prebuilt_image(
      accelerator_type=ctx.accelerator,
      base_image_repo=ctx.base_image_repo,
    )
    logging.info("Using prebuilt base image: %s", image_uri)
  elif ctx.container_image is None or ctx.container_image == "bundled":
    logging.info("Building container image...")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    image_uri = container_builder.get_or_build_container(
      base_image=f"python:{py_version}-slim",
      requirements_path=ctx.requirements_path,
      accelerator_type=ctx.accelerator,
      project=ctx.project,
      zone=ctx.zone,
      cluster_name=ctx.cluster_name,
    )
  else:
    assert ctx.container_image is not None
    image_uri = ctx.container_image
    logging.info("Using custom container: %s", image_uri)
  return image_uri


def _upload_artifacts(ctx: JobContext) -> bool:
  """Upload artifacts to Cloud Storage.

  Returns True if requirements content was resolved (prebuilt mode),
  False if requirements should be cleared.
  """
  if ctx.payload_path is None or ctx.context_path is None:
    raise ValueError("payload_path and context_path must be set before upload")
  logging.info("Uploading artifacts to Cloud Storage (job: %s)...", ctx.job_id)

  # In prebuilt mode, upload filtered requirements for runtime install.
  requirements_content = None
  has_requirements = True
  if _is_prebuilt(ctx):
    requirements_content = container_builder.prepare_requirements_content(
      ctx.requirements_path
    )
    if requirements_content is None:
      has_requirements = False

  storage.upload_artifacts(
    bucket_name=ctx.bucket_name,
    job_id=ctx.job_id,
    payload_path=ctx.payload_path,
    context_path=ctx.context_path,
    project=ctx.project,
    requirements_content=requirements_content,
  )
  return has_requirements


def _requirements_uri(ctx: JobContext) -> str | None:
  """Return the GCS URI for requirements.txt if prebuilt mode is active."""
  if _is_prebuilt(ctx) and ctx.requirements_path is not None:
    return f"gs://{ctx.bucket_name}/{ctx.job_id}/requirements.txt"
  return None


def prepare_execution(ctx: JobContext, backend: BaseK8sBackend) -> None:
  """Run the shared pre-submit phases for a remote job."""
  ensure_credentials(
    project=ctx.project,
    zone=ctx.zone,
    cluster=backend.cluster,
  )
  backend.validate_preflight(ctx)

  with tempfile.TemporaryDirectory() as tmpdir:
    _prepare_artifacts(ctx, tmpdir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
      build_future = pool.submit(_build_container, ctx)
      upload_future = pool.submit(_upload_artifacts, ctx)
      # Collect results on the main thread — avoid mutating ctx in workers.
      ctx.image_uri = build_future.result()
      has_requirements = upload_future.result()
      if not has_requirements:
        ctx.requirements_path = None


def submit_remote(ctx: JobContext, backend: BaseK8sBackend) -> JobHandle:
  """Submit a job and return a JobHandle without waiting for completion.

  Runs the shared pre-submit phases (credentials, preflight, prepare,
  build, upload), persists a durable handle to GCS, and submits the
  job to Kubernetes.  The caller observes, collects, and cleans up
  via the returned handle.

  Returns:
      A `JobHandle` representing the submitted job.
  """

  prepare_execution(ctx, backend)

  handle = JobHandle.from_job_context(
    ctx,
    backend_name=backend.name,
    namespace=backend.namespace,
    k8s_name=backend.get_k8s_name(ctx.job_id),
  )

  try:
    storage.upload_handle(
      ctx.bucket_name,
      ctx.job_id,
      handle.to_dict(),
      project=ctx.project,
    )
  except Exception:
    storage.cleanup_artifacts(ctx.bucket_name, ctx.job_id, project=ctx.project)
    raise

  try:
    backend.submit_job(ctx)
  except Exception as submit_error:
    try:
      if backend.job_exists(handle.k8s_name):
        logging.warning(
          "Kubernetes create for %s failed but resource %s exists; "
          "treating submission as successful",
          ctx.job_id,
          handle.k8s_name,
        )
        return handle
    except Exception:
      logging.warning(
        "Failed to reconcile submit error for job %s",
        ctx.job_id,
      )

    try:
      storage.cleanup_artifacts(
        ctx.bucket_name, ctx.job_id, project=ctx.project
      )
    except Exception:
      logging.warning("Failed to clean up GCS artifacts for job %s", ctx.job_id)
    raise submit_error from None

  return handle
