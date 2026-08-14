"""Docker-tier fixture: the real runner image, run the way GKE runs it.

Builds the container image through kinetic's own build machinery
(``_prepare_dockerfile`` + ``_pack_build_context``), so the build context
is byte-identical to what Cloud Build would consume, then executes it
with ``docker run`` using the exact command, args, and env that
``gke_client._create_job_spec`` stamps into the Job manifest — derived
from the spec at runtime, never hand-copied, so it cannot drift.

Artifacts move through a dedicated fake-gcs-server that advertises
``host.docker.internal`` so the containerized runner can reach it; the
host side seeds and reads emulator state over the raw JSON API.

Notes:
  - The image installs ``keras-kinetic==<current version>`` from PyPI
    (that is what the real Dockerfile does), so an unreleased version
    bump fails this tier's build exactly as it would fail Cloud Build.
  - The base image tracks the host interpreter's Python minor, also
    mirroring production behavior.
"""

import atexit
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest

from kinetic.backend import gke_client, k8s_utils
from kinetic.infra import container_builder
from kinetic.utils.fake_gcs_fixture import FakeGcsServer, binary_path

_IMAGE_REPO = "kinetic-test-runner"
_BUILD_TIMEOUT_SECONDS = 1200
_DOCKER_HOST_ALIAS = "host.docker.internal"

_docker_available = None
_runner_image = None
_server = None


def docker_available():
  """Whether a usable docker daemon is reachable (memoized)."""
  global _docker_available
  if _docker_available is None:
    if shutil.which("docker") is None:
      _docker_available = False
    else:
      try:
        _docker_available = (
          subprocess.run(
            ["docker", "info"], capture_output=True, timeout=30
          ).returncode
          == 0
        )
      except (subprocess.TimeoutExpired, OSError):
        _docker_available = False
  return _docker_available


def runner_image():
  """Build (or reuse) the runner image; returns its tag.

  The tag embeds a hash of the generated Dockerfile plus
  remote_runner.py — the same inputs Cloud Build would consume — so
  repeated runs reuse the docker-cached image until either changes.
  """
  global _runner_image
  if _runner_image is not None:
    return _runner_image

  with tempfile.TemporaryDirectory() as tmpdir:
    dockerfile_path = container_builder._prepare_dockerfile(tmpdir, "cpu", None)
    with open(dockerfile_path, "rb") as f:
      dockerfile_bytes = f.read()
    runner_path = os.path.join(
      container_builder._RUNNER_DIR, container_builder.REMOTE_RUNNER_FILE_NAME
    )
    with open(runner_path, "rb") as f:
      runner_bytes = f.read()
    content_hash = hashlib.sha256(
      dockerfile_bytes + b"\x00" + runner_bytes
    ).hexdigest()[:12]
    tag = f"{_IMAGE_REPO}:cpu-{content_hash}"

    inspect = subprocess.run(
      ["docker", "image", "inspect", tag], capture_output=True
    )
    if inspect.returncode != 0:
      tarball_path = container_builder._pack_build_context(
        tmpdir, dockerfile_path
      )
      with open(tarball_path, "rb") as tarball:
        build = subprocess.run(
          ["docker", "build", "-t", tag, "-"],
          stdin=tarball,
          capture_output=True,
          text=True,
          timeout=_BUILD_TIMEOUT_SECONDS,
        )
      if build.returncode != 0:
        raise RuntimeError(
          f"docker build of the runner image failed:\n{build.stderr[-4000:]}"
        )

  _runner_image = tag
  return tag


def emulator():
  """The docker tier's fake-gcs-server, advertising host.docker.internal."""
  global _server
  if _server is None:
    server = FakeGcsServer(advertised_host=_DOCKER_HOST_ALIAS)
    server.start()
    atexit.register(server.stop)
    _server = server
  return _server


# Container fields derive_docker_command translates into docker run,
# plus the ones that are Kubernetes-only concerns with no docker
# equivalent. Anything _create_job_spec sets outside this set fails
# loudly instead of being silently dropped.
_TRANSLATED_FIELDS = {
  "name",
  "image",
  "command",
  "args",
  "env",
  "volume_mounts",
}
_K8S_ONLY_FIELDS = {"resources"}


def _assert_spec_is_translatable(container):
  """Fail loudly when the Job spec grows a field this tier ignores."""
  for field, value in container.to_dict().items():
    if value is None or field in _TRANSLATED_FIELDS | _K8S_ONLY_FIELDS:
      continue
    raise RuntimeError(
      f"_create_job_spec set container field {field!r} which "
      "derive_docker_command does not translate. Teach it about the "
      "field (or allowlist it as k8s-only) so the docker tier stays "
      "faithful to the manifest."
    )
  for env in container.env or []:
    if env.value is None or env.value_from is not None:
      raise RuntimeError(
        f"env var {env.name!r} uses value_from/None, which docker run "
        "cannot express; derive_docker_command would silently garble it."
      )


def derive_docker_command(
  image,
  bucket_name,
  job_id,
  emulator_port,
  requirements_uri=None,
  payload_sha256=None,
  context_sha256=None,
  fuse_volume_specs=None,
  fuse_host_dirs=None,
):
  """Translate the real Job spec into an equivalent ``docker run``.

  The container command, args, env, and volume mounts come straight out
  of ``_create_job_spec``.  The Dockerfile template defines no
  ENTRYPOINT (asserted by the test suite), so passing command+args as
  the docker run command is exactly Kubernetes' command/args semantics.

  Args:
      fuse_volume_specs: Passed through to ``_create_job_spec`` so the
          real GCS-FUSE volumeMount machinery produces the mounts.
      fuse_host_dirs: ``{mount_path: host_dir}`` — the host directory
          standing in for each FUSE mount's bucket content.  Mount
          paths and read-only flags come from the spec's volumeMounts,
          never from the caller.
  """
  job = gke_client._create_job_spec(
    job_name=f"kinetic-{job_id}",
    container_uri=image,
    accel_config=k8s_utils.parse_accelerator("cpu"),
    job_id=job_id,
    bucket_name=bucket_name,
    namespace="default",
    requirements_uri=requirements_uri,
    fuse_volume_specs=fuse_volume_specs,
    payload_sha256=payload_sha256,
    context_sha256=context_sha256,
  )
  container = job.spec.template.spec.containers[0]
  _assert_spec_is_translatable(container)

  command = ["docker", "run", "--rm", "--name", f"kinetic-test-{job_id}"]
  # On Linux the host alias needs an explicit gateway mapping; Docker
  # Desktop (macOS/Windows) resolves it natively and ignores nothing —
  # the flag is accepted everywhere.
  command.append(f"--add-host={_DOCKER_HOST_ALIAS}:host-gateway")
  for env in container.env or []:
    command.extend(["-e", f"{env.name}={env.value}"])
  # The one deviation from the manifest: point the runner's storage
  # client at the emulator instead of Workload Identity + real GCS.
  command.extend(
    ["-e", f"STORAGE_EMULATOR_HOST=http://{_DOCKER_HOST_ALIAS}:{emulator_port}"]
  )
  for mount in container.volume_mounts or []:
    host_dir = (fuse_host_dirs or {}).get(mount.mount_path)
    if host_dir is None:
      raise RuntimeError(
        f"no host directory provided for spec volumeMount {mount.mount_path!r}"
      )
    mode = "ro" if mount.read_only else "rw"
    command.extend(["-v", f"{host_dir}:{mount.mount_path}:{mode}"])
  command.append(container.image)
  command.extend(container.command)
  command.extend(container.args)
  return command


class DockerTierTestCase(unittest.TestCase):
  """Base class: skips without docker or the emulator binary.

  With ``KINETIC_DOCKER_TIER_REQUIRED`` set (as CI sets it), missing
  prerequisites are an error instead of a skip, so the CI job can never
  silently go green as an all-skipped no-op.
  """

  image: str = None
  server: FakeGcsServer = None

  @classmethod
  def _unavailable(cls, reason):
    if os.environ.get("KINETIC_DOCKER_TIER_REQUIRED"):
      raise RuntimeError(f"{reason}, but KINETIC_DOCKER_TIER_REQUIRED is set")
    raise unittest.SkipTest(reason)

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if not docker_available():
      cls._unavailable("docker daemon not available")
    if binary_path() is None:
      cls._unavailable("fake-gcs-server binary not found")
    cls.server = emulator()
    cls.image = runner_image()
