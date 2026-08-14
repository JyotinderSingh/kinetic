"""Test fixture: a real fake-gcs-server behind STORAGE_EMULATOR_HOST.

Starts a `fake-gcs-server <https://github.com/fsouza/fake-gcs-server>`_
subprocess and points ``google-cloud-storage`` at it via
``STORAGE_EMULATOR_HOST``, so tests exercise the real GCS wire protocol —
uploads, downloads, 404/NotFound semantics, list pagination — without a
GCP account or network access.  This is the canonical transport for every
test that touches Cloud Storage; tests are skipped when the binary is
unavailable.  CI installs it; locally::

    brew install fake-gcs-server
    # or
    go install github.com/fsouza/fake-gcs-server@latest

or point ``FAKE_GCS_SERVER_BIN`` at the binary.
"""

import atexit
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid

from absl.testing import parameterized

_STARTUP_TIMEOUT_SECONDS = 15
_PROBE_INTERVAL_SECONDS = 0.05
# Applied to every JSON-API request so a wedged emulator fails the test
# quickly instead of hanging the suite.
_HTTP_TIMEOUT_SECONDS = 10

TEST_PROJECT = "test-project"

_shared_server = None


def binary_path():
  """Return the fake-gcs-server binary path, or None if unavailable."""
  override = os.environ.get("FAKE_GCS_SERVER_BIN")
  if override:
    return override if os.access(override, os.X_OK) else None
  return shutil.which("fake-gcs-server")


def skip_unless_fake_gcs(reason="fake-gcs-server binary not found"):
  """Skip decorator for tests requiring the fake-gcs-server binary."""
  return unittest.skipUnless(binary_path(), reason)


def shared_server():
  """Return the process-wide emulator, starting it on first use.

  Starting it exports ``STORAGE_EMULATOR_HOST`` (and a deterministic
  ``GOOGLE_CLOUD_PROJECT`` unless one is already set) for the rest of the
  process, and drops kinetic's cached storage clients so nothing keeps a
  client built for a different endpoint.  The server is stopped atexit.

  Raises:
      unittest.SkipTest: When the fake-gcs-server binary is unavailable.
  """
  global _shared_server
  if _shared_server is None:
    if os.environ.get("E2E_TESTS"):
      raise unittest.SkipTest(
        "emulator tests are skipped when E2E_TESTS is set: the emulator "
        "exports STORAGE_EMULATOR_HOST process-wide, which would redirect "
        "the real-GCS e2e suite. Run tests/e2e in a separate process."
      )
    if binary_path() is None:
      raise unittest.SkipTest("fake-gcs-server binary not found")
    server = FakeGcsServer()
    server.start()
    atexit.register(server.stop)
    os.environ["STORAGE_EMULATOR_HOST"] = server.host
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", TEST_PROJECT)
    clear_kinetic_client_cache()
    _shared_server = server
  return _shared_server


def clear_kinetic_client_cache():
  """Drop kinetic's cached storage clients (they pin an endpoint)."""
  from kinetic.utils import storage as kinetic_storage

  with kinetic_storage._client_lock:
    kinetic_storage._cached_clients.clear()


def _free_port():
  """Reserve and return a free localhost TCP port."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


class FakeGcsServer:
  """A fake-gcs-server subprocess bound to a free localhost port."""

  def __init__(self):
    self.port = None
    self._process = None
    self._stderr_file = None

  @property
  def host(self):
    """The emulator endpoint, suitable for STORAGE_EMULATOR_HOST."""
    return f"http://127.0.0.1:{self.port}"

  def start(self):
    """Start the server and wait until it answers HTTP."""
    binary = binary_path()
    if binary is None:
      raise RuntimeError("fake-gcs-server binary not found")
    self.port = _free_port()
    # The server logs every request to stderr; a PIPE would fill up and
    # block it mid-suite, so stderr goes to a temp file instead.  The
    # file outlives this method (closed in stop()), hence no `with`.
    self._stderr_file = tempfile.TemporaryFile()  # noqa: SIM115
    # -external-url / -public-host make resumable-upload session URLs and
    # object links point back at the emulator instead of
    # storage.googleapis.com.
    self._process = subprocess.Popen(
      [
        binary,
        "-scheme",
        "http",
        "-backend",
        "memory",
        "-port",
        str(self.port),
        "-external-url",
        self.host,
        "-public-host",
        f"127.0.0.1:{self.port}",
      ],
      stdout=subprocess.DEVNULL,
      stderr=self._stderr_file,
    )
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
      if self._process.poll() is not None:
        self._stderr_file.seek(0)
        stderr = self._stderr_file.read().decode(errors="replace")
        raise RuntimeError(
          f"fake-gcs-server exited with {self._process.returncode}: {stderr}"
        )
      try:
        with urllib.request.urlopen(
          f"{self.host}/storage/v1/b?project=_probe", timeout=1
        ):
          return
      except urllib.error.HTTPError:
        return  # Any HTTP status means the server is up.
      except (urllib.error.URLError, OSError):
        time.sleep(_PROBE_INTERVAL_SECONDS)
    self.stop()
    raise RuntimeError(
      f"fake-gcs-server did not become ready within "
      f"{_STARTUP_TIMEOUT_SECONDS}s on port {self.port}"
    )

  def stop(self):
    """Terminate the server subprocess and release its stderr file.

    Safe to call at any point of the lifecycle: a half-started server
    (Popen failed after the stderr file was created) still gets its
    file closed, and an already-dead process is not an error.
    """
    if self._process is not None:
      try:
        self._process.terminate()
        self._process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        try:
          self._process.kill()
          self._process.wait()
        except OSError:
          pass
      except OSError:
        pass
      finally:
        self._process = None
    if self._stderr_file is not None:
      self._stderr_file.close()
      self._stderr_file = None

  # ------------------------------------------------------------------
  # Raw JSON-API helpers, independent of google-cloud-storage, so tests
  # can seed and assert emulator state without going through the code
  # under test.
  # ------------------------------------------------------------------

  def create_bucket(self, bucket_name, project=TEST_PROJECT):
    """Create a bucket via the JSON API; existing buckets are fine."""
    request = urllib.request.Request(
      f"{self.host}/storage/v1/b?project={project}",
      data=json.dumps({"name": bucket_name}).encode(),
      headers={"Content-Type": "application/json"},
      method="POST",
    )
    try:
      with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS):
        pass
    except urllib.error.HTTPError as e:
      if e.code != 409:  # 409 Conflict: the bucket already exists.
        raise

  def make_bucket(self, suffix="jobs"):
    """Create and return a uniquely named bucket."""
    bucket_name = f"itest-{uuid.uuid4().hex[:12]}-{suffix}"
    self.create_bucket(bucket_name)
    return bucket_name

  def list_blob_names(self, bucket_name, prefix=None):
    """Return the names of all blobs in *bucket_name*."""
    url = f"{self.host}/storage/v1/b/{bucket_name}/o"
    if prefix:
      url += f"?prefix={urllib.parse.quote(prefix)}"
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_SECONDS) as response:
      listing = json.load(response)
    return [item["name"] for item in listing.get("items", [])]

  def blob_exists(self, bucket_name, blob_name):
    """Whether *blob_name* exists in *bucket_name*."""
    url = (
      f"{self.host}/storage/v1/b/{bucket_name}/o/"
      f"{urllib.parse.quote(blob_name, safe='')}"
    )
    try:
      with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_SECONDS):
        return True
    except urllib.error.HTTPError as e:
      if e.code == 404:
        return False
      raise

  def read_blob(self, bucket_name, blob_name):
    """Return the raw bytes of *blob_name*, or None when absent."""
    url = (
      f"{self.host}/storage/v1/b/{bucket_name}/o/"
      f"{urllib.parse.quote(blob_name, safe='')}?alt=media"
    )
    try:
      with urllib.request.urlopen(
        url, timeout=_HTTP_TIMEOUT_SECONDS
      ) as response:
        return response.read()
    except urllib.error.HTTPError as e:
      if e.code == 404:
        return None
      raise

  def write_blob(self, bucket_name, blob_name, data):
    """Upload raw bytes to *blob_name* via the JSON API."""
    if isinstance(data, str):
      data = data.encode()
    url = (
      f"{self.host}/upload/storage/v1/b/{bucket_name}/o"
      f"?uploadType=media&name={urllib.parse.quote(blob_name, safe='')}"
    )
    request = urllib.request.Request(
      url,
      data=data,
      headers={"Content-Type": "application/octet-stream"},
      method="POST",
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS):
      pass


class FakeGcsTestCase(parameterized.TestCase):
  """Base class for tests that talk to the shared emulator.

  ``setUpClass`` starts (or reuses) the process-wide emulator and
  isolates the class from any developer profile in ``~/.kinetic``.
  Each test should create its own bucket via ``make_bucket()``.
  """

  PROJECT = TEST_PROJECT

  server: FakeGcsServer = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.server = shared_server()  # Raises SkipTest when unavailable.
    cls._saved_profiles_file = os.environ.get("KINETIC_PROFILES_FILE")
    # A nonexistent path: the loader treats a missing file as "no
    # profiles" but would fail parsing an existing empty one.
    os.environ["KINETIC_PROFILES_FILE"] = os.path.join(
      tempfile.mkdtemp(prefix="kinetic-itest-"), "absent-profiles.json"
    )

  @classmethod
  def tearDownClass(cls):
    if cls._saved_profiles_file is None:
      os.environ.pop("KINETIC_PROFILES_FILE", None)
    else:
      os.environ["KINETIC_PROFILES_FILE"] = cls._saved_profiles_file
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    # Other test modules patch storage.Client and can leave a stale mock
    # in kinetic's per-project client cache; drop it before every test
    # (per-test, because xdist can interleave foreign tests between two
    # methods of one class).
    clear_kinetic_client_cache()

  def make_bucket(self, suffix="jobs"):
    """Create and return a unique bucket for this test."""
    return self.server.make_bucket(suffix=suffix)
