"""Tests for kinetic.collections — async collection orchestration."""

import json
import os
import tempfile
import threading
import time
from unittest import mock

from absl.testing import absltest, parameterized
from google.api_core import exceptions as google_exceptions

from kinetic import collections as kinetic_collections
from kinetic.collections import (
  BatchError,
  BatchHandle,
  attach_batch,
  map,
)
from kinetic.job_status import JobStatus
from kinetic.jobs import JobHandle

# Captured before any test patches `time.sleep`, which the collections
# module reaches through the shared `time` module object.
_REAL_SLEEP = time.sleep


def _tick(_seconds=0):
  """Stand-in for `time.sleep` that yields instead of honouring the delay.

  Poll loops that a test drives from another thread must keep turning
  without burning a core, so this sleeps for a token millisecond rather
  than returning instantly.
  """
  _REAL_SLEEP(0.001)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_handle(
  job_id="job-test",
  backend="gke",
  project="proj",
  cluster_name="cluster",
  bucket_name="proj-kn-cluster-jobs",
  group_id=None,
  group_kind=None,
  group_index=None,
):
  return JobHandle(
    job_id=job_id,
    backend=backend,
    project=project,
    cluster_name=cluster_name,
    zone="us-central1-a",
    namespace="default",
    bucket_name=bucket_name,
    k8s_name=f"kinetic-{job_id}",
    image_uri="image:tag",
    accelerator="cpu",
    func_name="train",
    display_name=f"kinetic-train-{job_id}",
    created_at="2026-03-28T10:00:00Z",
    group_id=group_id,
    group_kind=group_kind,
    group_index=group_index,
  )


def _make_batch_handle(n_jobs=3, submission_complete=True):
  """Create a BatchHandle with n real (non-mock) JobHandle objects."""
  jobs = [_make_handle(job_id=f"job-{i}") for i in range(n_jobs)]
  handle = BatchHandle(
    group_id="grp-test1234",
    name="test-batch",
    tags={"env": "test"},
    jobs=jobs,
    _bucket_name="proj-kn-cluster-jobs",
    _project="proj",
  )
  if submission_complete:
    handle._submission_complete.set()
  return handle


# ------------------------------------------------------------------
# BatchError
# ------------------------------------------------------------------


class TestBatchError(absltest.TestCase):
  def test_attributes(self):
    h = _make_handle()
    err = BatchError("grp-123", [h], [None, 42])
    self.assertEqual(err.group_id, "grp-123")
    self.assertEqual(err.failures, [h])
    self.assertEqual(err.partial_results, [None, 42])

  def test_message_format(self):
    err = BatchError("grp-abc", [_make_handle()], [None, None, 42])
    self.assertIn("1 of 3", str(err))
    self.assertIn("grp-abc", str(err))


# ------------------------------------------------------------------
# BatchHandle methods
# ------------------------------------------------------------------


class TestBatchHandle(absltest.TestCase):
  def test_statuses(self):
    handle = _make_batch_handle(3)
    with mock.patch.object(
      JobHandle,
      "status",
      side_effect=[JobStatus.RUNNING, JobStatus.SUCCEEDED, JobStatus.PENDING],
    ):
      result = handle.statuses()

    self.assertEqual(
      result,
      [
        (0, JobStatus.RUNNING),
        (1, JobStatus.SUCCEEDED),
        (2, JobStatus.PENDING),
      ],
    )

  def test_statuses_skips_none_jobs(self):
    handle = _make_batch_handle(2)
    handle.jobs.append(None)
    with mock.patch.object(
      JobHandle,
      "status",
      side_effect=[JobStatus.RUNNING, JobStatus.SUCCEEDED],
    ):
      result = handle.statuses()
    self.assertEqual(len(result), 2)

  def test_status_counts(self):
    handle = _make_batch_handle(3)
    with mock.patch.object(
      JobHandle,
      "status",
      side_effect=[JobStatus.RUNNING, JobStatus.SUCCEEDED, JobStatus.RUNNING],
    ):
      counts = handle.status_counts()

    self.assertEqual(counts, {"RUNNING": 2, "SUCCEEDED": 1})

  def test_wait_returns_when_all_terminal(self):
    handle = _make_batch_handle(2)
    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      handle.wait()

  def test_wait_timeout(self):
    handle = _make_batch_handle(1)
    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.RUNNING),
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch(
        "kinetic.collections.time.monotonic",
        side_effect=[0, 0, 0, 100],
      ),
      self.assertRaises(TimeoutError),
    ):
      handle.wait(timeout=10)

  def test_wait_raises_submission_error(self):
    handle = _make_batch_handle(1)
    handle._submission_error = RuntimeError("submit failed")
    with self.assertRaisesRegex(RuntimeError, "submit failed"):
      handle.wait()

  def test_results_returns_values(self):
    handle = _make_batch_handle(2)
    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(JobHandle, "result", side_effect=[10, 20]),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      result = handle.results()

    self.assertEqual(result, [10, 20])

  def test_results_raises_batch_error(self):
    handle = _make_batch_handle(2)
    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(
        JobHandle,
        "result",
        side_effect=[42, ValueError("boom")],
      ),
      mock.patch("kinetic.collections.time.sleep"),
      self.assertRaises(BatchError) as ctx,
    ):
      handle.results()

    self.assertEqual(len(ctx.exception.failures), 1)
    self.assertEqual(ctx.exception.partial_results[0], 42)

  def test_results_return_exceptions(self):
    handle = _make_batch_handle(2)
    err = ValueError("boom")
    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(JobHandle, "result", side_effect=[42, err]),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      result = handle.results(return_exceptions=True)

    self.assertEqual(result[0], 42)
    self.assertIs(result[1], err)

  def test_failures_only_includes_failed(self):
    handle = _make_batch_handle(3)
    with mock.patch.object(
      JobHandle,
      "status",
      side_effect=[
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.FAILED,
      ],
    ):
      failed = handle.failures()
    self.assertEqual(len(failed), 2)

  def test_failures_excludes_not_found(self):
    """NOT_FOUND (e.g. after cleanup) should not be treated as failure."""
    handle = _make_batch_handle(3)
    with mock.patch.object(
      JobHandle,
      "status",
      side_effect=[
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.NOT_FOUND,
      ],
    ):
      failed = handle.failures()
    # Only the FAILED job, not the NOT_FOUND one.
    self.assertEqual(len(failed), 1)

  def test_cancel(self):
    handle = _make_batch_handle(3)
    with (
      mock.patch.object(
        JobHandle,
        "status",
        side_effect=[
          JobStatus.RUNNING,
          JobStatus.SUCCEEDED,
          JobStatus.RUNNING,
        ],
      ),
      mock.patch.object(JobHandle, "cancel") as mock_cancel,
    ):
      handle.cancel()

    self.assertEqual(mock_cancel.call_count, 2)

  def test_cleanup_delegates_to_jobs(self):
    handle = _make_batch_handle(2)
    with (
      mock.patch.object(JobHandle, "cleanup") as mock_cleanup,
      mock.patch(
        "kinetic.collections.storage.cleanup_manifest"
      ) as mock_manifest,
    ):
      handle.cleanup()

    self.assertEqual(mock_cleanup.call_count, 2)
    mock_manifest.assert_called_once_with(
      "proj-kn-cluster-jobs", "grp-test1234", project="proj"
    )

  def test_cleanup_k8s_only_skips_manifest(self):
    handle = _make_batch_handle(2)
    with (
      mock.patch.object(JobHandle, "cleanup") as mock_cleanup,
      mock.patch(
        "kinetic.collections.storage.cleanup_manifest"
      ) as mock_manifest,
    ):
      handle.cleanup(k8s=True, gcs=False)

    self.assertEqual(mock_cleanup.call_count, 2)
    mock_cleanup.assert_called_with(k8s=True, gcs=False)
    mock_manifest.assert_not_called()

  def test_results_unordered_returns_completion_order(self):
    """ordered=False should yield results in completion order."""
    handle = _make_batch_handle(2)

    # Simulate: job-1 finishes before job-0.
    status_calls = {0: 0, 1: 0}

    def mock_status(self_handle):
      idx = int(self_handle.job_id.split("-")[1])
      status_calls[idx] += 1
      if idx == 1:
        return JobStatus.SUCCEEDED
      # job-0 takes 2 polls to complete.
      if status_calls[0] >= 2:
        return JobStatus.SUCCEEDED
      return JobStatus.RUNNING

    def mock_result(self_handle, cleanup=True):
      idx = int(self_handle.job_id.split("-")[1])
      return f"result-{idx}"

    with (
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "result", mock_result),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      # Use ordered=False to get completion order.
      results = handle.results(ordered=False, cleanup=False)

    # job-1 completes first (immediate SUCCEEDED), job-0 takes 2 polls.
    self.assertEqual(len(results), 2)
    self.assertEqual(results[0], "result-1")
    self.assertEqual(results[1], "result-0")

  def test_as_completed_yields_before_submission_complete(self):
    """as_completed() must yield jobs that finish while submission is
    still in progress — not block until _submission_complete is set."""
    handle = _make_batch_handle(n_jobs=3, submission_complete=False)

    # Simulate: job-0 is already terminal, job-1 is running,
    # job-2 hasn't been submitted yet (None).
    handle.jobs[2] = None  # not yet submitted

    poll_round = [0]

    def mock_status(self_handle):
      idx = int(self_handle.job_id.split("-")[1])
      if idx == 0:
        return JobStatus.SUCCEEDED
      # job-1 and job-2 become terminal after poll_round advances.
      if poll_round[0] >= 1:
        return JobStatus.SUCCEEDED
      return JobStatus.RUNNING

    yielded = []

    def collect():
      for job in handle.as_completed(poll_interval=0.01, timeout=5):
        yielded.append(job.job_id)
        # After first yield, simulate the submission thread filling
        # slot 2 and advancing the poll round so remaining jobs finish.
        if len(yielded) == 1:
          handle.jobs[2] = _make_handle(job_id="job-2")
          poll_round[0] += 1
        if len(yielded) == 2:
          handle._submission_complete.set()

    with mock.patch.object(JobHandle, "status", mock_status):
      collect()

    # job-0 was yielded first (before submission_complete was set).
    self.assertEqual(yielded[0], "job-0")
    self.assertIn("job-1", yielded)
    self.assertEqual(len(yielded), 3)

  def test_as_completed_timeout_raises(self):
    """as_completed(timeout=...) must raise TimeoutError if jobs don't
    finish in time."""
    handle = _make_batch_handle(n_jobs=1, submission_complete=True)

    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.RUNNING),
      mock.patch(
        "kinetic.collections.time.monotonic",
        side_effect=[0, 0, 100],
      ),
      mock.patch("kinetic.collections.time.sleep"),
      self.assertRaises(TimeoutError),
    ):
      list(handle.as_completed(timeout=5))


class TestMap(absltest.TestCase):
  def _make_submit_fn(self, handles=None):
    """Return a mock submit_fn returning pre-built handles."""
    if handles is None:
      call_count = [0]

      def submit_fn(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        return _make_handle(job_id=f"job-{idx}")

      submit_fn.__name__ = "mock_train"
      return submit_fn

    handles_iter = iter(handles)

    def submit_fn(*args, **kwargs):
      return next(handles_iter)

    submit_fn.__name__ = "mock_train"
    return submit_fn

  def test_rejects_non_callable(self):
    with self.assertRaises(TypeError):
      map("not-a-function", [1, 2])

  def test_rejects_invalid_input_mode(self):
    with self.assertRaises(ValueError):
      map(lambda x: x, [1], input_mode="bogus")

  def test_rejects_empty_inputs(self):
    with self.assertRaises(ValueError):
      map(lambda x: x, [])

  def test_basic_map_submits_all_jobs(self):
    submit_fn = self._make_submit_fn()
    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
    ):
      handle = map(
        submit_fn,
        [1, 2, 3],
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )

    self.assertEqual(len(handle.jobs), 3)
    self.assertTrue(all(j is not None for j in handle.jobs))
    self.assertTrue(handle._submission_complete.is_set())

  def test_group_fields_set_on_handles(self):
    submit_fn = self._make_submit_fn()
    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
    ):
      handle = map(
        submit_fn,
        ["a", "b"],
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )

    for i, job in enumerate(handle.jobs):
      self.assertEqual(job.group_id, handle.group_id)
      self.assertEqual(job.group_kind, "map")
      self.assertEqual(job.group_index, i)

  def test_group_id_format(self):
    submit_fn = self._make_submit_fn()
    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
    ):
      handle = map(
        submit_fn,
        [1],
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )
    self.assertTrue(handle.group_id.startswith("grp-"))
    self.assertEqual(len(handle.group_id), 12)  # "grp-" + 8 hex

  def test_initial_manifest_uploaded_before_jobs(self):
    """The initial empty manifest should be uploaded before the first job."""
    call_order = []

    def track_manifest(*args, **kwargs):
      call_order.append("manifest")

    def track_handle(*args, **kwargs):
      call_order.append("handle")

    submit_fn = self._make_submit_fn()
    with (
      mock.patch(
        "kinetic.collections.storage.upload_manifest",
        side_effect=track_manifest,
      ),
      mock.patch(
        "kinetic.collections.storage.upload_handle",
        side_effect=track_handle,
      ),
    ):
      map(
        submit_fn,
        [1],
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )

    # First call should be manifest (initial empty), then handle
    # re-upload, then manifest update.
    self.assertEqual(call_order[0], "manifest")

  def test_name_and_tags_propagated(self):
    submit_fn = self._make_submit_fn()
    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
    ):
      handle = map(
        submit_fn,
        [1],
        name="my-batch",
        tags={"env": "test"},
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )

    self.assertEqual(handle.name, "my-batch")
    self.assertEqual(handle.tags, {"env": "test"})

  def test_max_concurrent_uses_background_thread(self):
    """When max_concurrent is set, submission runs in a background thread."""
    submit_fn = self._make_submit_fn()
    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
    ):
      handle = map(
        submit_fn,
        [1, 2],
        max_concurrent=10,
        project="proj",
        cluster="cluster",
      )
      # Wait for background thread to finish.
      handle._submission_complete.wait(timeout=5)

    self.assertEqual(len([j for j in handle.jobs if j is not None]), 2)

  def test_fail_fast_stops_submission(self):
    """When a job fails and fail_fast=True, remaining jobs are not submitted."""
    call_count = [0]

    def failing_submit(*args, **kwargs):
      idx = call_count[0]
      call_count[0] += 1
      if idx == 1:
        raise RuntimeError("submission failed")
      return _make_handle(job_id=f"job-{idx}")

    failing_submit.__name__ = "failing"

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      handle = map(
        failing_submit,
        [1, 2, 3],
        max_concurrent=None,
        fail_fast=True,
        project="proj",
        cluster="cluster",
      )

    # Job 0 submitted, job 1 failed, job 2 not submitted.
    submitted = [j for j in handle.jobs if j is not None]
    self.assertEqual(len(submitted), 1)

  def test_retry_resubmits_after_failure(self):
    handles_returned = []

    def make_submit():
      call_count = [0]

      def submit_fn(x):
        idx = call_count[0]
        call_count[0] += 1
        h = _make_handle(job_id=f"job-{idx}")
        handles_returned.append(h)
        return h

      submit_fn.__name__ = "fn"
      return submit_fn

    submit_fn = make_submit()

    # First call returns a handle that will be FAILED,
    # second call (retry) returns one that will be SUCCEEDED.
    status_calls = [0]

    def mock_status(self_handle):
      status_calls[0] += 1
      # The first handle (job-0) always reports FAILED.
      # The second handle (job-1) always reports SUCCEEDED.
      if self_handle.job_id == "job-0":
        return JobStatus.FAILED
      return JobStatus.SUCCEEDED

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "cleanup"),
    ):
      handle = map(
        submit_fn,
        [42],
        max_concurrent=1,
        retries=1,
        project="proj",
        cluster="cluster",
      )
      handle._submission_complete.wait(timeout=5)

    # submit_fn called twice: initial + 1 retry.
    self.assertEqual(len(handles_returned), 2)
    # Final job on the handle is the retry.
    self.assertEqual(handle.jobs[0].job_id, "job-1")

  def test_no_retry_when_attempts_exhausted(self):
    call_count = [0]

    def submit_fn(x):
      idx = call_count[0]
      call_count[0] += 1
      return _make_handle(job_id=f"job-{idx}")

    submit_fn.__name__ = "fn"

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch.object(JobHandle, "status", return_value=JobStatus.FAILED),
      mock.patch.object(JobHandle, "cleanup"),
    ):
      handle = map(
        submit_fn,
        [1],
        max_concurrent=1,
        retries=0,
        project="proj",
        cluster="cluster",
      )
      handle._submission_complete.wait(timeout=5)

    # Only 1 submission, no retries.
    self.assertEqual(call_count[0], 1)

  def test_submission_thread_is_not_daemon(self):
    """The background submission thread must be non-daemon so it survives
    interpreter shutdown and finishes submitting all jobs."""

    def submit_fn(x):
      return _make_handle(job_id=f"job-{x}")

    submit_fn.__name__ = "fn"

    started_threads = []
    original_start = threading.Thread.start

    def capture_start(self_thread):
      started_threads.append(self_thread)
      original_start(self_thread)

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(threading.Thread, "start", capture_start),
    ):
      handle = map(
        submit_fn,
        [1, 2],
        max_concurrent=1,
        project="proj",
        cluster="cluster",
      )
      handle._submission_complete.wait(timeout=5)

    self.assertEqual(len(started_threads), 1)
    self.assertFalse(started_threads[0].daemon)

  def test_cancel_siblings_on_runtime_failure(self):
    """When all jobs are submitted and one fails at runtime,
    fail_fast + cancel_running_on_fail must cancel siblings —
    even when there are no pending jobs left to submit."""
    call_count = [0]

    def submit_fn(x):
      idx = call_count[0]
      call_count[0] += 1
      return _make_handle(job_id=f"job-{idx}")

    submit_fn.__name__ = "fn"

    cancelled = set()
    poll_counts = {}

    def mock_status(self_handle):
      jid = self_handle.job_id
      poll_counts[jid] = poll_counts.get(jid, 0) + 1
      if jid == "job-0":
        if poll_counts[jid] >= 2:
          return JobStatus.FAILED
        return JobStatus.RUNNING
      if jid in cancelled:
        return JobStatus.NOT_FOUND
      return JobStatus.RUNNING

    def track_cancel(self_handle):
      cancelled.add(self_handle.job_id)

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "cancel", track_cancel),
    ):
      handle = map(
        submit_fn,
        [1, 2],
        max_concurrent=2,
        retries=0,
        fail_fast=True,
        cancel_running_on_fail=True,
        project="proj",
        cluster="cluster",
      )
      handle._submission_complete.wait(timeout=5)

    self.assertIn("job-1", cancelled)

  def test_zero_max_concurrent_rejected(self):
    with self.assertRaisesRegex(ValueError, "max_concurrent"):
      map(lambda x: x, [1], max_concurrent=0)

  def test_none_max_concurrent_synchronous(self):
    """max_concurrent=None (unlimited) submits synchronously."""

    def submit_fn(x):
      return _make_handle(job_id="job-0")

    submit_fn.__name__ = "fn"

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
    ):
      handle = map(
        submit_fn,
        [1],
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )
    self.assertIsNotNone(handle.jobs[0])
    self.assertTrue(handle._submission_complete.is_set())

  def test_max_concurrent_limits_active_jobs(self):
    """With max_concurrent=2, at most 2 jobs should be active at any
    point during the submission loop."""
    max_active_seen = [0]
    currently_active = [0]
    lock = threading.Lock()

    def submit_fn(x):
      with lock:
        currently_active[0] += 1
        if currently_active[0] > max_active_seen[0]:
          max_active_seen[0] = currently_active[0]
      return _make_handle(job_id=f"job-{x}")

    submit_fn.__name__ = "fn"

    poll_counts = {}
    decremented = set()

    def mock_status(self_handle):
      jid = self_handle.job_id
      poll_counts[jid] = poll_counts.get(jid, 0) + 1
      if poll_counts[jid] >= 2:
        with lock:
          if jid not in decremented:
            decremented.add(jid)
            currently_active[0] -= 1
        return JobStatus.SUCCEEDED
      return JobStatus.RUNNING

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch.object(JobHandle, "status", mock_status),
    ):
      handle = map(
        submit_fn,
        [0, 1, 2, 3, 4],
        max_concurrent=2,
        project="proj",
        cluster="cluster",
      )
      handle._submission_complete.wait(timeout=10)

    self.assertEqual(len([j for j in handle.jobs if j is not None]), 5)
    self.assertLessEqual(max_active_seen[0], 2)


class TestAttachBatch(absltest.TestCase):
  def _make_manifest(self, n_children=2, total_expected=None):
    if total_expected is None:
      total_expected = n_children
    return {
      "group_id": "grp-abc12345",
      "group_kind": "map",
      "group_name": "test-batch",
      "tags": {"env": "test"},
      "created_at": "2026-03-28T10:00:00Z",
      "total_expected": total_expected,
      "submit_fn_name": "train",
      "children": [
        {"group_index": i, "job_id": f"job-{i}", "attempts": 1}
        for i in range(n_children)
      ],
    }

  def _make_handle_payload(self, job_id):
    return {
      "job_id": job_id,
      "backend": "gke",
      "project": "proj",
      "cluster_name": "cluster",
      "zone": "us-central1-a",
      "namespace": "default",
      "bucket_name": "proj-kn-cluster-jobs",
      "k8s_name": f"kinetic-{job_id}",
      "image_uri": "image:tag",
      "accelerator": "cpu",
      "func_name": "train",
      "display_name": f"kinetic-train-{job_id}",
      "created_at": "2026-03-28T10:00:00Z",
      "group_id": "grp-abc12345",
      "group_kind": "map",
      "group_index": 0,
    }

  def _stage_profile(self, **fields):
    """Write a one-profile store and return env dict pointing at it."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="kinetic-profiles-")
    os.close(fd)
    self.addCleanup(os.unlink, path)
    payload = {"current": "p", "profiles": {"p": fields}}
    with open(path, "w", encoding="utf-8") as f:
      json.dump(payload, f)
    return {"KINETIC_PROFILES_FILE": path}

  def test_downloads_manifest_and_handles(self):
    manifest = self._make_manifest(2)

    def download_handle_side_effect(bucket, job_id, project=None):
      return self._make_handle_payload(job_id)

    with (
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        return_value=manifest,
      ),
      mock.patch(
        "kinetic.collections.storage.download_handle",
        side_effect=download_handle_side_effect,
      ),
    ):
      handle = attach_batch("grp-abc12345", project="proj", cluster="cluster")

    self.assertEqual(handle.group_id, "grp-abc12345")
    self.assertEqual(handle.name, "test-batch")
    self.assertEqual(len(handle.jobs), 2)
    self.assertTrue(handle._submission_complete.is_set())

  def test_attach_batch_falls_back_to_on_disk_profile(self):
    """attach_batch() reads the active profile when no env vars / kwargs set."""
    env = self._stage_profile(
      project="prof-proj",
      zone="prof-zone",
      cluster="prof-cluster",
      namespace="prof-ns",
    )
    manifest = self._make_manifest(1)

    with (
      mock.patch.dict(os.environ, env, clear=True),
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        return_value=manifest,
      ) as mock_download_manifest,
      mock.patch(
        "kinetic.collections.storage.download_handle",
        return_value=self._make_handle_payload("job-0"),
      ),
    ):
      handle = attach_batch("grp-abc12345")

    self.assertEqual(handle.group_id, "grp-abc12345")
    mock_download_manifest.assert_called_once_with(
      "prof-proj-kn-prof-cluster-jobs",
      "grp-abc12345",
      project="prof-proj",
    )

  def test_missing_child_handle_preserves_index(self):
    """Missing handle.json should leave a None at the correct index."""
    manifest = self._make_manifest(2)

    def download_side_effect(bucket, job_id, project=None):
      if job_id == "job-1":
        raise google_exceptions.NotFound("gone")
      return self._make_handle_payload(job_id)

    with (
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        return_value=manifest,
      ),
      mock.patch(
        "kinetic.collections.storage.download_handle",
        side_effect=download_side_effect,
      ),
    ):
      handle = attach_batch("grp-abc12345", project="proj", cluster="cluster")

    # List still has 2 entries (total_expected=2), index 1 is None.
    self.assertEqual(len(handle.jobs), 2)
    self.assertEqual(handle.jobs[0].job_id, "job-0")
    self.assertIsNone(handle.jobs[1])

  def test_attach_preserves_group_index_with_gaps(self):
    """Children at non-contiguous indices should land at their group_index."""
    manifest = {
      "group_id": "grp-abc12345",
      "group_kind": "map",
      "group_name": "test-batch",
      "tags": {},
      "created_at": "2026-03-28T10:00:00Z",
      "total_expected": 4,
      "submit_fn_name": "train",
      "children": [
        {"group_index": 0, "job_id": "job-0", "attempts": 1},
        {"group_index": 3, "job_id": "job-3", "attempts": 1},
      ],
    }

    with (
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        return_value=manifest,
      ),
      mock.patch(
        "kinetic.collections.storage.download_handle",
        return_value=self._make_handle_payload("job-0"),
      ),
    ):
      handle = attach_batch("grp-abc12345", project="proj", cluster="cluster")

    self.assertEqual(len(handle.jobs), 4)
    self.assertIsNotNone(handle.jobs[0])
    self.assertIsNone(handle.jobs[1])
    self.assertIsNone(handle.jobs[2])
    self.assertIsNotNone(handle.jobs[3])


class TestSubmissionErrors(absltest.TestCase):
  def test_submission_error_surfaced_in_results(self):
    """Per-input submission failures should raise BatchError."""
    call_count = [0]

    def failing_submit(*args, **kwargs):
      idx = call_count[0]
      call_count[0] += 1
      if idx == 1:
        raise RuntimeError("bad input")
      return _make_handle(job_id=f"job-{idx}")

    failing_submit.__name__ = "fn"

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(JobHandle, "result", return_value=42),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      handle = map(
        failing_submit,
        [1, 2, 3],
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )
      with self.assertRaises(BatchError) as ctx:
        handle.results()

    # Index 1 failed at submission time.  No job exists for it, so it is
    # reported through submission_failures and never lands in failures.
    self.assertEqual(ctx.exception.failures, [])
    self.assertIsInstance(ctx.exception.submission_failures[1], RuntimeError)

  def test_submission_error_with_return_exceptions(self):
    """Per-input submission failures should appear as exceptions."""
    call_count = [0]

    def failing_submit(*args, **kwargs):
      idx = call_count[0]
      call_count[0] += 1
      if idx == 1:
        raise RuntimeError("bad input")
      return _make_handle(job_id=f"job-{idx}")

    failing_submit.__name__ = "fn"

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(JobHandle, "result", return_value=42),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      handle = map(
        failing_submit,
        [1, 2, 3],
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )
      results = handle.results(return_exceptions=True)

    self.assertEqual(results[0], 42)
    self.assertIsInstance(results[1], RuntimeError)
    self.assertEqual(results[2], 42)


# ------------------------------------------------------------------
# JobHandle group field serialization (regression)
# ------------------------------------------------------------------


class TestJobHandleGroupFields(absltest.TestCase):
  def test_to_dict_omits_none_group_fields(self):
    h = _make_handle()
    d = h.to_dict()
    self.assertNotIn("group_id", d)
    self.assertNotIn("group_kind", d)
    self.assertNotIn("group_index", d)

  def test_to_dict_includes_set_group_fields(self):
    h = _make_handle(group_id="grp-1", group_kind="map", group_index=5)
    d = h.to_dict()
    self.assertEqual(d["group_id"], "grp-1")
    self.assertEqual(d["group_kind"], "map")
    self.assertEqual(d["group_index"], 5)

  def test_from_dict_round_trip_with_group_fields(self):
    h = _make_handle(group_id="grp-1", group_kind="map", group_index=3)
    d = h.to_dict()
    rebuilt = JobHandle.from_dict(d)
    self.assertEqual(rebuilt.group_id, "grp-1")
    self.assertEqual(rebuilt.group_kind, "map")
    self.assertEqual(rebuilt.group_index, 3)

  def test_from_dict_without_group_fields_defaults_to_none(self):
    d = {
      "job_id": "job-1",
      "backend": "gke",
      "project": "proj",
      "cluster_name": "cluster",
      "zone": "us-central1-a",
      "namespace": "default",
      "bucket_name": "bucket",
      "k8s_name": "kinetic-job-1",
      "image_uri": "image:tag",
      "accelerator": "cpu",
      "func_name": "train",
      "display_name": "display",
      "created_at": "2026-03-28T10:00:00Z",
    }
    h = JobHandle.from_dict(d)
    self.assertIsNone(h.group_id)
    self.assertIsNone(h.group_kind)
    self.assertIsNone(h.group_index)


class TestSubmissionFailuresProperty(absltest.TestCase):
  def test_returns_copy_of_errors(self):
    handle = _make_batch_handle(3)
    handle._submission_errors = {1: RuntimeError("bad input")}
    result = handle.submission_failures
    self.assertEqual(len(result), 1)
    self.assertIsInstance(result[1], RuntimeError)
    # Mutating the returned dict must not affect internal state.
    result[99] = ValueError("injected")
    self.assertNotIn(99, handle._submission_errors)

  def test_empty_when_no_errors(self):
    handle = _make_batch_handle(2)
    self.assertEqual(handle.submission_failures, {})


class TestWaitWarnsOnSubmissionErrors(absltest.TestCase):
  def test_warns_when_submission_errors_present(self):
    handle = _make_batch_handle(2)
    handle._submission_errors = {1: RuntimeError("bad")}
    # Only job-0 is real; job-1 slot is None.
    handle.jobs[1] = None
    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch("kinetic.collections.logging") as mock_logging,
    ):
      handle.wait()
    mock_logging.warning.assert_called_once()
    # logging.warning receives the format string and args separately.
    fmt = mock_logging.warning.call_args[0][0]
    self.assertIn("input(s) failed at submission time", fmt)


class TestFailuresCaching(absltest.TestCase):
  def test_failures_cached_after_results_cleanup(self):
    """After results(cleanup=True), failures() should return cached failures."""
    handle = _make_batch_handle(2)

    def mock_status(self_handle):
      return JobStatus.SUCCEEDED

    def mock_result(self_handle, cleanup=True):
      idx = int(self_handle.job_id.split("-")[1])
      if idx == 1:
        raise ValueError("job-1 failed")
      return 42

    with (
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "result", mock_result),
      mock.patch("kinetic.collections.time.sleep"),
      self.assertRaises(BatchError),
    ):
      handle.results(cleanup=True)

    # After cleanup, live status would be NOT_FOUND, but cached
    # failures should still report the failed job.
    with mock.patch.object(
      JobHandle, "status", return_value=JobStatus.NOT_FOUND
    ):
      failed = handle.failures()
    self.assertEqual(len(failed), 1)
    self.assertEqual(failed[0].job_id, "job-1")

  def test_failures_live_when_no_results_called(self):
    """Without calling results(), failures() should check live status."""
    handle = _make_batch_handle(3)
    with mock.patch.object(
      JobHandle,
      "status",
      side_effect=[
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.SUCCEEDED,
      ],
    ):
      failed = handle.failures()
    self.assertEqual(len(failed), 1)

  def test_failures_cache_is_copy(self):
    """Modifying the list from failures() must not affect the cache."""
    handle = _make_batch_handle(2)

    def mock_result(self_handle, cleanup=True):
      raise ValueError("boom")

    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(JobHandle, "result", mock_result),
      mock.patch("kinetic.collections.time.sleep"),
      self.assertRaises(BatchError),
    ):
      handle.results()

    first_call = handle.failures()
    first_call.clear()
    second_call = handle.failures()
    self.assertEqual(len(second_call), 2)

  def test_failures_with_return_exceptions(self):
    """failures() should return failed jobs even when return_exceptions=True."""
    handle = _make_batch_handle(2)

    def mock_status(self_handle):
      return JobStatus.SUCCEEDED

    def mock_result(self_handle, cleanup=True):
      idx = int(self_handle.job_id.split("-")[1])
      if idx == 1:
        raise ValueError("job-1 failed")
      return 42

    with (
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "result", mock_result),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      results = handle.results(return_exceptions=True, cleanup=False)

    self.assertEqual(results[0], 42)
    self.assertIsInstance(results[1], ValueError)

    failed = handle.failures()
    self.assertEqual(len(failed), 1)
    self.assertEqual(failed[0].job_id, "job-1")


class TestAttachBatchPolling(absltest.TestCase):
  def _make_handle_payload(self, job_id, group_index=0):
    return {
      "job_id": job_id,
      "backend": "gke",
      "project": "proj",
      "cluster_name": "cluster",
      "zone": "us-central1-a",
      "namespace": "default",
      "bucket_name": "proj-kn-cluster-jobs",
      "k8s_name": f"kinetic-{job_id}",
      "image_uri": "image:tag",
      "accelerator": "cpu",
      "func_name": "train",
      "display_name": f"kinetic-train-{job_id}",
      "created_at": "2026-03-28T10:00:00Z",
      "group_id": "grp-abc12345",
      "group_kind": "map",
      "group_index": group_index,
    }

  def test_complete_batch_no_polling(self):
    """When all children are present, no background thread is started."""
    manifest = {
      "group_id": "grp-abc12345",
      "group_kind": "map",
      "group_name": "test",
      "tags": {},
      "created_at": "2026-03-28T10:00:00Z",
      "total_expected": 2,
      "submit_fn_name": "train",
      "children": [
        {"group_index": 0, "job_id": "job-0", "attempts": 1},
        {"group_index": 1, "job_id": "job-1", "attempts": 1},
      ],
    }

    def download_handle_side_effect(bucket, job_id, project=None):
      idx = int(job_id.split("-")[1])
      return self._make_handle_payload(job_id, group_index=idx)

    with (
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        return_value=manifest,
      ),
      mock.patch(
        "kinetic.collections.storage.download_handle",
        side_effect=download_handle_side_effect,
      ),
    ):
      handle = attach_batch("grp-abc12345", project="proj", cluster="cluster")

    self.assertTrue(handle._submission_complete.is_set())
    self.assertEqual(len([j for j in handle.jobs if j is not None]), 2)

  def test_partial_batch_polls_manifest(self):
    """Partial batch should poll manifest until all children appear."""
    partial_manifest = {
      "group_id": "grp-abc12345",
      "group_kind": "map",
      "group_name": "test",
      "tags": {},
      "created_at": "2026-03-28T10:00:00Z",
      "total_expected": 3,
      "submit_fn_name": "train",
      "children": [
        {"group_index": 0, "job_id": "job-0", "attempts": 1},
      ],
    }
    full_manifest = {
      **partial_manifest,
      "children": [
        {"group_index": 0, "job_id": "job-0", "attempts": 1},
        {"group_index": 1, "job_id": "job-1", "attempts": 1},
        {"group_index": 2, "job_id": "job-2", "attempts": 1},
      ],
    }

    # Gate to hold the poll loop until we've checked the initial state.
    gate = threading.Event()
    download_manifest_calls = [0]

    def download_manifest_side_effect(bucket, group_id, project=None):
      download_manifest_calls[0] += 1
      if download_manifest_calls[0] <= 1:
        # Initial call from attach_batch().
        return partial_manifest
      # Poll-loop calls: wait for the gate, then return full manifest.
      gate.wait(timeout=5)
      return full_manifest

    def download_handle_side_effect(bucket, job_id, project=None):
      idx = int(job_id.split("-")[1])
      return self._make_handle_payload(job_id, group_index=idx)

    with (
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        side_effect=download_manifest_side_effect,
      ),
      mock.patch(
        "kinetic.collections.storage.download_handle",
        side_effect=download_handle_side_effect,
      ),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      handle = attach_batch(
        "grp-abc12345",
        project="proj",
        cluster="cluster",
        poll_interval=0.01,
      )
      # Should NOT be set immediately for partial batch.
      self.assertFalse(handle._submission_complete.is_set())
      # Release the poll loop.
      gate.set()
      # Wait for the background poll thread to complete.
      handle._submission_complete.wait(timeout=5)

    self.assertTrue(handle._submission_complete.is_set())
    self.assertEqual(len([j for j in handle.jobs if j is not None]), 3)

  def test_partial_batch_timeout(self):
    """Polling should stop and set _submission_complete on timeout."""
    partial_manifest = {
      "group_id": "grp-abc12345",
      "group_kind": "map",
      "group_name": "test",
      "tags": {},
      "created_at": "2026-03-28T10:00:00Z",
      "total_expected": 3,
      "submit_fn_name": "train",
      "children": [
        {"group_index": 0, "job_id": "job-0", "attempts": 1},
      ],
    }

    def download_handle_side_effect(bucket, job_id, project=None):
      idx = int(job_id.split("-")[1])
      return self._make_handle_payload(job_id, group_index=idx)

    # Use real monotonic time to let the timeout elapse.
    with (
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        return_value=partial_manifest,
      ),
      mock.patch(
        "kinetic.collections.storage.download_handle",
        side_effect=download_handle_side_effect,
      ),
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch(
        "kinetic.collections.time.monotonic",
        side_effect=[0, 100],
      ),
    ):
      handle = attach_batch(
        "grp-abc12345",
        project="proj",
        cluster="cluster",
        poll_interval=0.01,
        poll_timeout=0.05,
      )
      handle._submission_complete.wait(timeout=5)

    # _submission_complete set even on timeout (so wait() doesn't hang).
    self.assertTrue(handle._submission_complete.is_set())
    # Only 1 of 3 loaded.
    self.assertEqual(len([j for j in handle.jobs if j is not None]), 1)

  def test_poll_loop_skips_invalid_group_index(self):
    """Poll loop must skip children with missing/non-int/out-of-range indices.

    Regression: previously the loop used ``child.get("group_index", -1)``
    which both aliased missing indices to ``jobs[-1]`` and raised
    ``IndexError`` for out-of-range indices, crashing the poll thread.
    """
    partial_manifest = {
      "group_id": "grp-abc12345",
      "group_kind": "map",
      "group_name": "test",
      "tags": {},
      "created_at": "2026-03-28T10:00:00Z",
      "total_expected": 2,
      "submit_fn_name": "train",
      "children": [
        {"group_index": 0, "job_id": "job-0", "attempts": 1},
      ],
    }
    # Poll manifest contains one valid child (index 1) plus junk entries
    # that used to break the loop.
    poll_manifest = {
      **partial_manifest,
      "children": [
        {"group_index": 0, "job_id": "job-0", "attempts": 1},
        {"job_id": "job-missing-idx", "attempts": 1},  # no group_index
        {"group_index": "bad", "job_id": "job-str-idx", "attempts": 1},
        {"group_index": 99, "job_id": "job-oob", "attempts": 1},
        {"group_index": 1, "job_id": "job-1", "attempts": 1},
      ],
    }

    manifest_calls = [0]

    def download_manifest_side_effect(bucket, group_id, project=None):
      manifest_calls[0] += 1
      if manifest_calls[0] <= 1:
        return partial_manifest
      return poll_manifest

    def download_handle_side_effect(bucket, job_id, project=None):
      idx = int(job_id.split("-")[1])
      return self._make_handle_payload(job_id, group_index=idx)

    with (
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        side_effect=download_manifest_side_effect,
      ),
      mock.patch(
        "kinetic.collections.storage.download_handle",
        side_effect=download_handle_side_effect,
      ),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      handle = attach_batch(
        "grp-abc12345",
        project="proj",
        cluster="cluster",
        poll_interval=0.01,
      )
      handle._submission_complete.wait(timeout=5)

    # Poll thread survived the junk entries and loaded the valid child.
    self.assertTrue(handle._submission_complete.is_set())
    self.assertIsNotNone(handle.jobs[0])
    self.assertIsNotNone(handle.jobs[1])


# ------------------------------------------------------------------
# cancel() — stops launching, and cancelled jobs are never retried
# ------------------------------------------------------------------


class TestCancelStopsLaunching(absltest.TestCase):
  """`cancel()` must cover queued inputs, not only running jobs."""

  def _submit_fn(self, submitted, first_submitted):
    def submit_fn(x):
      submitted.append(x)
      first_submitted.set()
      return _make_handle(job_id=f"job-{x}")

    submit_fn.__name__ = "fn"
    return submit_fn

  def test_cancel_drops_queued_inputs(self):
    """With bounded concurrency, queued inputs must not launch after cancel."""
    submitted = []
    first_submitted = threading.Event()
    cancelled = set()

    def mock_status(self_handle):
      if self_handle.job_id in cancelled:
        return JobStatus.NOT_FOUND
      return JobStatus.RUNNING

    def track_cancel(self_handle):
      cancelled.add(self_handle.job_id)

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep", _tick),
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "cancel", track_cancel),
    ):
      handle = map(
        self._submit_fn(submitted, first_submitted),
        [0, 1, 2],
        max_concurrent=1,
        project="proj",
        cluster="cluster",
      )
      self.assertTrue(first_submitted.wait(timeout=5))
      handle.cancel()
      self.assertTrue(handle._submission_complete.wait(timeout=5))

    # Only the job that was already running got submitted; indices 1 and
    # 2 were still queued behind max_concurrent=1 and are dropped.
    self.assertEqual(submitted, [0])
    self.assertEqual(cancelled, {"job-0"})
    self.assertIsNone(handle.jobs[1])
    self.assertIsNone(handle.jobs[2])

  def test_cancel_does_not_resubmit_with_retries(self):
    """Cancelling turns a job NOT_FOUND; retries must not resurrect it."""
    submitted = []
    first_submitted = threading.Event()
    cancelled = set()

    def mock_status(self_handle):
      if self_handle.job_id in cancelled:
        return JobStatus.NOT_FOUND
      return JobStatus.RUNNING

    def track_cancel(self_handle):
      cancelled.add(self_handle.job_id)

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep", _tick),
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "cancel", track_cancel),
      mock.patch.object(JobHandle, "cleanup") as mock_cleanup,
    ):
      handle = map(
        self._submit_fn(submitted, first_submitted),
        [0],
        max_concurrent=1,
        retries=2,
        project="proj",
        cluster="cluster",
      )
      self.assertTrue(first_submitted.wait(timeout=5))
      handle.cancel()
      self.assertTrue(handle._submission_complete.wait(timeout=5))

    self.assertEqual(submitted, [0])
    # The retry path cleans up the previous attempt's K8s resources
    # before it re-queues the index.  Nothing may go down that path, or
    # the loop is still treating the cancelled job as a failed attempt.
    mock_cleanup.assert_not_called()

  def test_cancel_catches_job_launched_during_cancel(self):
    """A slot filled after cancel() snapshotted it must still be cancelled.

    `cancel()` sets its flag before reading `handle.jobs`, so a
    submission already in flight lands after the snapshot.  Closing that
    window is the submission loop's job.
    """
    entered_second = threading.Event()
    release_second = threading.Event()

    def submit_fn(x):
      if x == 1:
        entered_second.set()
        self.assertTrue(release_second.wait(timeout=5))
      return _make_handle(job_id=f"job-{x}")

    submit_fn.__name__ = "fn"

    cancelled = set()

    def mock_status(self_handle):
      if self_handle.job_id in cancelled:
        return JobStatus.NOT_FOUND
      return JobStatus.RUNNING

    def track_cancel(self_handle):
      cancelled.add(self_handle.job_id)

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep", _tick),
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "cancel", track_cancel),
    ):
      handle = map(
        submit_fn,
        [0, 1],
        max_concurrent=2,
        project="proj",
        cluster="cluster",
      )
      self.assertTrue(entered_second.wait(timeout=5))
      # job-1 has not been registered yet, so cancel() cannot see it.
      handle.cancel()
      release_second.set()
      self.assertTrue(handle._submission_complete.wait(timeout=5))

    self.assertEqual(cancelled, {"job-0", "job-1"})

  def test_cancel_lets_as_completed_finish(self):
    """Slots cancelled before launch must not stall as_completed()."""
    handle = _make_batch_handle(n_jobs=1, submission_complete=False)
    handle.jobs.extend([None, None])

    cancelled = set()

    def mock_status(self_handle):
      if self_handle.job_id in cancelled:
        return JobStatus.NOT_FOUND
      return JobStatus.RUNNING

    def track_cancel(self_handle):
      cancelled.add(self_handle.job_id)

    with (
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "cancel", track_cancel),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      handle.cancel()
      # Stands in for the submission loop, which this test does not run.
      handle._submission_complete.set()
      yielded = list(handle.as_completed(poll_interval=0.01, timeout=5))

    self.assertEqual(cancelled, {"job-0"})
    # The two slots that never held a job must not stall the iterator.
    self.assertEqual([j.job_id for j in yielded], ["job-0"])


# ------------------------------------------------------------------
# attach_batch() after the children were cleaned up
# ------------------------------------------------------------------


class TestAttachAfterChildCleanup(absltest.TestCase):
  """`results(cleanup=True)` deletes each child's whole GCS prefix.

  The group manifest survives, so a later `attach_batch()` sees children
  it can name but whose `handle.json` is gone.  Those slots are terminal,
  not pending — polling for them would block `wait()` forever.
  """

  def _manifest(self, n_children=2, total_expected=None):
    if total_expected is None:
      total_expected = n_children
    return {
      "group_id": "grp-abc12345",
      "group_kind": "map",
      "group_name": "test-batch",
      "tags": {},
      "created_at": "2026-03-28T10:00:00Z",
      "total_expected": total_expected,
      "submit_fn_name": "train",
      "children": [
        {"group_index": i, "job_id": f"job-{i}", "attempts": 1}
        for i in range(n_children)
      ],
    }

  def _attach(self, manifest, **kwargs):
    with (
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        return_value=manifest,
      ),
      mock.patch(
        "kinetic.collections.storage.download_handle",
        side_effect=google_exceptions.NotFound("artifacts deleted"),
      ),
    ):
      return attach_batch(
        "grp-abc12345", project="proj", cluster="cluster", **kwargs
      )

  def test_completes_without_starting_a_poll_thread(self):
    with mock.patch.object(threading.Thread, "start") as mock_start:
      handle = self._attach(self._manifest(2))

    mock_start.assert_not_called()
    self.assertTrue(handle._submission_complete.is_set())
    self.assertEqual(handle.jobs, [None, None])

  def test_unavailable_children_names_the_missing_jobs(self):
    handle = self._attach(self._manifest(2))
    self.assertEqual(handle.unavailable_children, {0: "job-0", 1: "job-1"})
    # The property hands back a copy, not the live mapping.
    handle.unavailable_children[9] = "job-9"
    self.assertNotIn(9, handle.unavailable_children)

  def test_wait_returns_instead_of_blocking(self):
    handle = self._attach(self._manifest(2))
    with mock.patch("kinetic.collections.time.sleep") as mock_sleep:
      handle.wait(timeout=5)
    mock_sleep.assert_not_called()

  def test_results_return_none_without_blocking(self):
    handle = self._attach(self._manifest(2))
    with mock.patch("kinetic.collections.time.sleep"):
      results = handle.results(timeout=5)
    self.assertEqual(results, [None, None])

  def test_results_warns_about_unavailable_children(self):
    handle = self._attach(self._manifest(2))
    with (
      mock.patch("kinetic.collections.time.sleep"),
      mock.patch("kinetic.collections.logging") as mock_logging,
    ):
      handle.results(timeout=5)

    warnings = [c[0][0] for c in mock_logging.warning.call_args_list]
    self.assertTrue(
      any("have no handle in GCS" in w for w in warnings),
      f"expected an unavailable-children warning, got {warnings}",
    )

  def test_as_completed_terminates(self):
    handle = self._attach(self._manifest(2))
    with mock.patch("kinetic.collections.time.sleep"):
      yielded = list(handle.as_completed(poll_interval=0.01, timeout=5))
    self.assertEqual(yielded, [])

  def test_genuinely_partial_batch_still_polls(self):
    """A manifest short of total_expected is still worth waiting on."""
    with mock.patch.object(threading.Thread, "start") as mock_start:
      handle = self._attach(self._manifest(n_children=1, total_expected=3))

    mock_start.assert_called_once()
    self.assertFalse(handle._submission_complete.is_set())
    # Cleared so the daemon thread that never started cannot strand
    # anything waiting on this handle.
    handle._submission_complete.set()

  def test_poll_thread_gets_a_bounded_default_timeout(self):
    recorded = {}

    def fake_poll_loop(**kwargs):
      recorded.update(kwargs)
      kwargs["handle"]._submission_complete.set()

    with mock.patch("kinetic.collections._manifest_poll_loop", fake_poll_loop):
      handle = self._attach(self._manifest(n_children=1, total_expected=3))
      self.assertTrue(handle._submission_complete.wait(timeout=5))

    self.assertIsNotNone(recorded["timeout"])
    self.assertEqual(
      recorded["timeout"],
      kinetic_collections._DEFAULT_MANIFEST_POLL_TIMEOUT,
    )

  def test_poll_loop_stops_once_every_child_is_named(self):
    """Polling ends on manifest coverage, not on handles that load."""
    partial = self._manifest(n_children=1, total_expected=2)
    full = self._manifest(n_children=2, total_expected=2)

    manifest_calls = [0]

    def download_manifest_side_effect(bucket, group_id, project=None):
      manifest_calls[0] += 1
      return partial if manifest_calls[0] <= 1 else full

    def download_handle_side_effect(bucket, job_id, project=None):
      if job_id == "job-1":
        raise google_exceptions.NotFound("artifacts deleted")
      return {
        "job_id": job_id,
        "backend": "gke",
        "project": "proj",
        "cluster_name": "cluster",
        "zone": "us-central1-a",
        "namespace": "default",
        "bucket_name": "proj-kn-cluster-jobs",
        "k8s_name": f"kinetic-{job_id}",
        "image_uri": "image:tag",
        "accelerator": "cpu",
        "func_name": "train",
        "display_name": f"kinetic-train-{job_id}",
        "created_at": "2026-03-28T10:00:00Z",
        "group_id": "grp-abc12345",
        "group_kind": "map",
        "group_index": 0,
      }

    with (
      mock.patch(
        "kinetic.collections.storage.download_manifest",
        side_effect=download_manifest_side_effect,
      ),
      mock.patch(
        "kinetic.collections.storage.download_handle",
        side_effect=download_handle_side_effect,
      ),
      mock.patch("kinetic.collections.time.sleep", _tick),
    ):
      handle = attach_batch(
        "grp-abc12345",
        project="proj",
        cluster="cluster",
        poll_interval=0.01,
      )
      self.assertTrue(handle._submission_complete.wait(timeout=5))

    self.assertIsNotNone(handle.jobs[0])
    self.assertIsNone(handle.jobs[1])
    self.assertEqual(handle.unavailable_children, {1: "job-1"})


# ------------------------------------------------------------------
# BatchError — failures hold real handles, submission errors are separate
# ------------------------------------------------------------------


class TestBatchErrorFailureReporting(absltest.TestCase):
  def _mixed_batch_handle(self):
    """A batch with one submission error and one runtime failure."""

    def failing_submit(x):
      if x == 1:
        raise RuntimeError("bad input")
      return _make_handle(job_id=f"job-{x}")

    failing_submit.__name__ = "fn"

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
    ):
      return map(
        failing_submit,
        [0, 1, 2],
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )

  def _collect(self, handle, **kwargs):
    def mock_result(self_handle, cleanup=True):
      if self_handle.job_id == "job-2":
        raise ValueError("job-2 failed at runtime")
      return 42

    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(JobHandle, "result", mock_result),
      mock.patch("kinetic.collections.time.sleep"),
    ):
      return handle.results(**kwargs)

  def test_documented_failure_loop_does_not_raise(self):
    """`for job in e.failures: job.job_id` is documented — it must work."""
    handle = self._mixed_batch_handle()
    with self.assertRaises(BatchError) as ctx:
      self._collect(handle, cleanup=False)

    job_ids = [job.job_id for job in ctx.exception.failures]
    self.assertEqual(job_ids, ["job-2"])

  def test_submission_errors_reach_the_caller(self):
    handle = self._mixed_batch_handle()
    with self.assertRaises(BatchError) as ctx:
      self._collect(handle, cleanup=False)

    err = ctx.exception
    self.assertEqual(sorted(err.submission_failures), [1])
    self.assertIsInstance(err.submission_failures[1], RuntimeError)
    # Both kinds of failure are counted in the message.
    self.assertIn("2 of 3", str(err))

  def test_submission_error_alone_still_raises(self):
    """A batch whose jobs all succeed but whose input failed must raise."""

    def failing_submit(x):
      if x == 1:
        raise RuntimeError("bad input")
      return _make_handle(job_id=f"job-{x}")

    failing_submit.__name__ = "fn"

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
    ):
      handle = map(
        failing_submit,
        [0, 1],
        max_concurrent=None,
        project="proj",
        cluster="cluster",
      )

    with (
      mock.patch.object(JobHandle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(JobHandle, "result", return_value=42),
      mock.patch("kinetic.collections.time.sleep"),
      self.assertRaises(BatchError) as ctx,
    ):
      handle.results(cleanup=False)

    self.assertEqual(ctx.exception.failures, [])
    self.assertIsInstance(ctx.exception.submission_failures[1], RuntimeError)

  def test_completion_order_failures_are_handles_too(self):
    handle = self._mixed_batch_handle()
    with self.assertRaises(BatchError) as ctx:
      self._collect(handle, ordered=False, cleanup=False)

    # Named explicitly: an empty list would satisfy an all() check while
    # hiding the completion-order path dropping failures altogether.
    self.assertEqual([job.job_id for job in ctx.exception.failures], ["job-2"])
    self.assertIsInstance(ctx.exception.submission_failures[1], RuntimeError)

  def test_handle_exposes_the_same_submission_failures(self):
    handle = self._mixed_batch_handle()
    self.assertIsInstance(handle.submission_failures[1], RuntimeError)

  def test_submission_failures_default_to_empty(self):
    """The three-argument constructor stays valid."""
    err = BatchError("grp-123", [_make_handle()], [None, 42])
    self.assertEqual(err.submission_failures, {})

  def test_submission_failures_are_copied(self):
    source = {1: RuntimeError("bad input")}
    err = BatchError("grp-123", [], [None, None], source)
    source[2] = ValueError("injected")
    self.assertEqual(sorted(err.submission_failures), [1])


# ------------------------------------------------------------------
# Threading decision — map() must hand back a handle, not block
# ------------------------------------------------------------------


class TestRunsInCallingThread(parameterized.TestCase):
  @parameterized.named_parameters(
    # (max_concurrent, retries, fail_fast, cancel_running_on_fail, expected)
    ("plain", None, 0, False, False, True),
    ("fail_fast_only", None, 0, True, False, True),
    ("cancel_flag_without_fail_fast", None, 0, False, True, True),
    ("fail_fast_and_cancel", None, 0, True, True, False),
    ("bounded_concurrency", 8, 0, False, False, False),
    ("retries", None, 1, False, False, False),
    ("bounded_with_fail_fast", 8, 0, True, False, False),
  )
  def test_decision(
    self, max_concurrent, retries, fail_fast, cancel_running_on_fail, expected
  ):
    self.assertEqual(
      kinetic_collections._runs_in_calling_thread(
        max_concurrent, retries, fail_fast, cancel_running_on_fail
      ),
      expected,
    )

  def test_fail_fast_returns_a_handle_without_waiting(self):
    """fail_fast alone must not make map() wait for the jobs to finish.

    With everything submitted, no retries, and no siblings to cancel, a
    terminal status changes nothing the loop could act on — so the loop
    has no reason to poll, and map() has no reason to block.
    """
    polled = []

    def submit_fn(x):
      return _make_handle(job_id=f"job-{x}")

    submit_fn.__name__ = "fn"

    def mock_status(self_handle):
      polled.append(self_handle.job_id)
      return JobStatus.SUCCEEDED

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep") as mock_sleep,
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(threading.Thread, "start") as mock_start,
    ):
      handle = map(
        submit_fn,
        [0, 1, 2],
        max_concurrent=None,
        retries=0,
        fail_fast=True,
        project="proj",
        cluster="cluster",
      )

    self.assertTrue(handle._submission_complete.is_set())
    self.assertEqual(len([j for j in handle.jobs if j is not None]), 3)
    # No background thread, and no waiting on the jobs it just launched.
    mock_start.assert_not_called()
    mock_sleep.assert_not_called()
    self.assertEqual(polled, [])

  def test_fail_fast_with_cancel_keeps_the_background_thread(self):
    """Cancelling siblings needs someone watching for the first failure."""

    def submit_fn(x):
      return _make_handle(job_id=f"job-{x}")

    submit_fn.__name__ = "fn"

    cancelled = set()

    def mock_status(self_handle):
      if self_handle.job_id == "job-0":
        return JobStatus.FAILED
      if self_handle.job_id in cancelled:
        return JobStatus.NOT_FOUND
      return JobStatus.RUNNING

    def track_cancel(self_handle):
      cancelled.add(self_handle.job_id)

    started_threads = []
    original_start = threading.Thread.start

    def capture_start(self_thread):
      started_threads.append(self_thread)
      original_start(self_thread)

    with (
      mock.patch("kinetic.collections.storage.upload_manifest"),
      mock.patch("kinetic.collections.storage.upload_handle"),
      mock.patch("kinetic.collections.time.sleep", _tick),
      mock.patch.object(JobHandle, "status", mock_status),
      mock.patch.object(JobHandle, "cancel", track_cancel),
      mock.patch.object(threading.Thread, "start", capture_start),
    ):
      handle = map(
        submit_fn,
        [0, 1],
        max_concurrent=None,
        retries=0,
        fail_fast=True,
        cancel_running_on_fail=True,
        project="proj",
        cluster="cluster",
      )
      self.assertTrue(handle._submission_complete.wait(timeout=5))

    self.assertEqual(cancelled, {"job-1"})
    # The cancellation above can only happen off the calling thread.
    self.assertEqual(len(started_threads), 1)
    self.assertFalse(started_threads[0].daemon)


if __name__ == "__main__":
  absltest.main()
