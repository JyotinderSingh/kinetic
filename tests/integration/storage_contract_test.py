"""Client-side GCS contract tests against a real fake-gcs-server.

These exercise the storage semantics that unit tests hand-fake with Mock
blobs: real 404/NotFound classes from the wire, real list pagination,
real transfer_manager parallel transfers, the data-cache marker
protocol, and the timing-dependent result-download backoff in
``JobHandle``.  Assertions read emulator state directly, never log text.
"""

import os
import pathlib
import tempfile
import threading
import time
from unittest import mock

import cloudpickle
from absl.testing import absltest
from google.cloud import exceptions as cloud_exceptions

from kinetic import jobs
from kinetic.constants import build_bucket_name
from kinetic.data import Data
from kinetic.utils import storage
from kinetic.utils.fake_gcs_fixture import FakeGcsTestCase


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


def _write_artifacts(tmp_path):
  """Write a local payload.pkl and context.zip pair; returns their paths."""
  payload_path = tmp_path / "payload.pkl"
  payload_path.write_bytes(b"payload bytes")
  context_path = tmp_path / "context.zip"
  context_path.write_bytes(b"context bytes")
  return str(payload_path), str(context_path)


def _make_handle(bucket_name, job_id="job-itest01"):
  """A JobHandle aimed at *bucket_name*; k8s fields are inert strings."""
  return jobs.JobHandle(
    job_id=job_id,
    backend="gke",
    project=FakeGcsTestCase.PROJECT,
    cluster_name="itest-cluster",
    zone="us-central1-a",
    namespace="default",
    bucket_name=bucket_name,
    k8s_name=f"kinetic-{job_id}",
    image_uri="img",
    accelerator="cpu",
    func_name="fn",
    display_name="fn",
    created_at="2026-01-01T00:00:00Z",
  )


class TestUploadArtifacts(FakeGcsTestCase):
  def test_uploads_payload_context_and_requirements(self):
    bucket = self.make_bucket()
    payload_path, context_path = _write_artifacts(_make_temp_path(self))

    storage.upload_artifacts(
      bucket,
      "job-1",
      payload_path,
      context_path,
      project=self.PROJECT,
      requirements_content="numpy==1.26\n",
    )

    self.assertEqual(
      self.server.read_blob(bucket, "job-1/payload.pkl"), b"payload bytes"
    )
    self.assertEqual(
      self.server.read_blob(bucket, "job-1/context.zip"), b"context bytes"
    )
    self.assertEqual(
      self.server.read_blob(bucket, "job-1/requirements.txt"),
      b"numpy==1.26\n",
    )

  def test_requirements_omitted_when_not_provided(self):
    bucket = self.make_bucket()
    payload_path, context_path = _write_artifacts(_make_temp_path(self))

    storage.upload_artifacts(
      bucket, "job-2", payload_path, context_path, project=self.PROJECT
    )

    self.assertFalse(self.server.blob_exists(bucket, "job-2/requirements.txt"))


class TestDownloadResult(FakeGcsTestCase):
  def test_downloads_to_a_local_file(self):
    bucket = self.make_bucket()
    self.server.write_blob(bucket, "job-3/result.pkl", b"result bytes")

    local_path = storage.download_result(bucket, "job-3", project=self.PROJECT)
    self.addCleanup(os.remove, local_path)

    self.assertEqual(pathlib.Path(local_path).read_bytes(), b"result bytes")

  def test_missing_result_raises_the_real_not_found(self):
    bucket = self.make_bucket()

    with self.assertRaises(cloud_exceptions.NotFound):
      storage.download_result(bucket, "job-none", project=self.PROJECT)


class TestHandleRoundtrip(FakeGcsTestCase):
  def test_upload_then_download(self):
    bucket = self.make_bucket()
    payload = {"job_id": "job-4", "backend": "gke", "bucket_name": bucket}

    storage.upload_handle(bucket, "job-4", payload, project=self.PROJECT)

    self.assertEqual(
      storage.download_handle(bucket, "job-4", project=self.PROJECT), payload
    )

  def test_missing_handle_raises_not_found(self):
    bucket = self.make_bucket()

    with self.assertRaises(cloud_exceptions.NotFound):
      storage.download_handle(bucket, "job-none", project=self.PROJECT)


class TestManifests(FakeGcsTestCase):
  def test_roundtrip_and_cleanup(self):
    bucket = self.make_bucket()
    manifest = {"kind": "batch", "children": ["job-a", "job-b"]}

    storage.upload_manifest(bucket, "grp-1", manifest, project=self.PROJECT)
    self.assertEqual(
      storage.download_manifest(bucket, "grp-1", project=self.PROJECT),
      manifest,
    )

    storage.cleanup_manifest(bucket, "grp-1", project=self.PROJECT)
    self.assertEqual(self.server.list_blob_names(bucket, "_groups/"), [])

  def test_missing_manifest_raises_file_not_found(self):
    bucket = self.make_bucket()

    # The documented contract maps GCS NotFound to FileNotFoundError.
    with self.assertRaises(FileNotFoundError):
      storage.download_manifest(bucket, "grp-none", project=self.PROJECT)


class TestCleanupArtifacts(FakeGcsTestCase):
  def test_deletes_only_the_job_prefix(self):
    bucket = self.make_bucket()
    for name in ("job-5/payload.pkl", "job-5/result.pkl", "job-50/keep.pkl"):
      self.server.write_blob(bucket, name, b"x")

    storage.cleanup_artifacts(bucket, "job-5", project=self.PROJECT)

    self.assertEqual(self.server.list_blob_names(bucket, "job-5/"), [])
    self.assertTrue(self.server.blob_exists(bucket, "job-50/keep.pkl"))

  def test_empty_prefix_is_a_noop(self):
    bucket = self.make_bucket()

    storage.cleanup_artifacts(bucket, "job-none", project=self.PROJECT)


class TestBlobHelpers(FakeGcsTestCase):
  def test_upload_empty_blob_then_exists(self):
    bucket = self.make_bucket()

    self.assertFalse(self.server.blob_exists(bucket, "markers/m1"))
    storage.upload_empty_blob(bucket, "markers/m1", project=self.PROJECT)

    self.assertTrue(storage.blob_exists(bucket, "markers/m1", self.PROJECT))
    self.assertEqual(self.server.read_blob(bucket, "markers/m1"), b"")


class TestUploadDataCache(FakeGcsTestCase):
  """The content-addressed data cache and its marker-sentinel protocol."""

  def _local_file_data(self, content=b"training bytes"):
    path = _make_temp_path(self) / "train.csv"
    path.write_bytes(content)
    return Data(str(path))

  def test_cache_miss_uploads_content_and_marker(self):
    bucket = self.make_bucket()
    data = self._local_file_data()
    content_hash = data.content_hash()

    uri = storage.upload_data(bucket, data, project=self.PROJECT)

    cache_prefix = f"default/data-cache/{content_hash}"
    self.assertEqual(uri, f"gs://{bucket}/{cache_prefix}")
    self.assertEqual(
      self.server.read_blob(bucket, f"{cache_prefix}/train.csv"),
      b"training bytes",
    )
    self.assertTrue(
      self.server.blob_exists(bucket, f"default/data-markers/{content_hash}")
    )

  def test_cache_hit_short_circuits_on_the_marker_alone(self):
    bucket = self.make_bucket()
    data = self._local_file_data()
    content_hash = data.content_hash()
    # Only the marker exists; the cached content is deliberately absent.
    # A hit must trust the marker and skip the upload entirely.
    self.server.write_blob(bucket, f"default/data-markers/{content_hash}", b"")

    uri = storage.upload_data(bucket, data, project=self.PROJECT)

    self.assertEqual(uri, f"gs://{bucket}/default/data-cache/{content_hash}")
    self.assertEqual(
      self.server.list_blob_names(bucket, "default/data-cache/"), []
    )

  def test_directory_upload_preserves_structure(self):
    bucket = self.make_bucket()
    data_dir = _make_temp_path(self) / "dataset"
    (data_dir / "sub").mkdir(parents=True)
    (data_dir / "train.csv").write_bytes(b"t")
    (data_dir / "sub" / "eval.csv").write_bytes(b"e")
    data = Data(str(data_dir))

    uri = storage.upload_data(bucket, data, project=self.PROJECT)

    prefix = uri.removeprefix(f"gs://{bucket}/")
    self.assertEqual(
      sorted(self.server.list_blob_names(bucket, prefix)),
      [f"{prefix}/sub/eval.csv", f"{prefix}/train.csv"],
    )

  def test_gcs_uri_data_is_passed_through_untouched(self):
    bucket = self.make_bucket()

    uri = storage.upload_data(
      bucket, Data("gs://elsewhere/dataset"), project=self.PROJECT
    )

    self.assertEqual(uri, "gs://elsewhere/dataset")
    self.assertEqual(self.server.list_blob_names(bucket), [])

  def test_namespace_prefix_scopes_the_cache(self):
    bucket = self.make_bucket()
    data = self._local_file_data()

    uri = storage.upload_data(
      bucket, data, project=self.PROJECT, namespace_prefix="team-a"
    )

    self.assertTrue(
      uri.startswith(f"gs://{bucket}/team-a/data-cache/"),
      uri,
    )
    self.assertTrue(
      self.server.blob_exists(
        bucket, f"team-a/data-markers/{data.content_hash()}"
      )
    )


class TestJobHandleResultBackoff(FakeGcsTestCase):
  """jobs.py's NotFound-retry loop against real 404s and real timing."""

  def test_retries_until_the_result_appears(self):
    bucket = self.make_bucket()
    handle = _make_handle(bucket)
    result = {"success": True, "result": 5}

    def seed_late():
      time.sleep(1.2)
      self.server.write_blob(
        bucket, f"{handle.job_id}/result.pkl", cloudpickle.dumps(result)
      )

    seeder = threading.Thread(target=seed_late)
    seeder.start()
    self.addCleanup(seeder.join)

    payload = handle._download_result_payload_with_backoff(deadline=None)

    self.assertEqual(payload, result)

  def test_missing_result_raises_not_found_within_the_deadline(self):
    bucket = self.make_bucket()
    handle = _make_handle(bucket)

    with self.assertRaises(cloud_exceptions.NotFound):
      handle._download_result_payload_with_backoff(
        deadline=time.monotonic() + 0.5
      )

  def test_undeserializable_result_keeps_the_artifacts(self):
    bucket = self.make_bucket()
    handle = _make_handle(bucket)
    self.server.write_blob(
      bucket, f"{handle.job_id}/result.pkl", b"not a pickle"
    )

    with self.assertRaisesRegex(RuntimeError, "Could not deserialize"):
      handle._download_result_payload()

    self.assertTrue(
      self.server.blob_exists(bucket, f"{handle.job_id}/result.pkl")
    )


class TestAttachHydration(FakeGcsTestCase):
  """attach() rebuilds a handle purely from GCS state — no cluster calls."""

  def _conventional_bucket(self, cluster="itest-cluster"):
    bucket = build_bucket_name(self.PROJECT, cluster)
    self.server.create_bucket(bucket)
    return bucket

  def test_attach_hydrates_from_handle_json(self):
    bucket = self._conventional_bucket()
    source = _make_handle(bucket, job_id="job-attach1")
    storage.upload_handle(
      bucket, source.job_id, source.to_dict(), project=self.PROJECT
    )

    with mock.patch.dict(
      os.environ,
      {"KINETIC_PROJECT": self.PROJECT, "KINETIC_CLUSTER": "itest-cluster"},
    ):
      handle = jobs.attach("job-attach1")

    self.assertEqual(handle, source)

  def test_attach_ignores_unknown_fields_from_future_versions(self):
    bucket = self._conventional_bucket()
    source = _make_handle(bucket, job_id="job-attach2")
    payload = {**source.to_dict(), "field_from_the_future": "ignored"}
    storage.upload_handle(bucket, source.job_id, payload, project=self.PROJECT)

    with mock.patch.dict(
      os.environ,
      {"KINETIC_PROJECT": self.PROJECT, "KINETIC_CLUSTER": "itest-cluster"},
    ):
      handle = jobs.attach("job-attach2")

    self.assertEqual(handle.job_id, "job-attach2")

  def test_attach_missing_handle_raises_not_found(self):
    self._conventional_bucket()

    with (
      mock.patch.dict(
        os.environ,
        {"KINETIC_PROJECT": self.PROJECT, "KINETIC_CLUSTER": "itest-cluster"},
      ),
      self.assertRaises(cloud_exceptions.NotFound),
    ):
      jobs.attach("job-none")


class TestCleanupGcsOnly(FakeGcsTestCase):
  def test_cleanup_without_k8s_deletes_artifacts_only(self):
    bucket = self.make_bucket()
    handle = _make_handle(bucket)
    for name in ("payload.pkl", "context.zip", "result.pkl", "handle.json"):
      self.server.write_blob(bucket, f"{handle.job_id}/{name}", b"x")

    handle.cleanup(k8s=False, gcs=True)

    self.assertEqual(
      self.server.list_blob_names(bucket, f"{handle.job_id}/"), []
    )


if __name__ == "__main__":
  absltest.main()
