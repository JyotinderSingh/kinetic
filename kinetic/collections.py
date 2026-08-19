"""Async collection orchestration for Kinetic.

Provides `map()` for job-array-style fan-out, `BatchHandle` for
observing and collecting collection results, and `attach_batch()`
for cross-session reattachment.
"""

from __future__ import annotations

import collections
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from absl import logging
from google.api_core import exceptions as google_exceptions

from kinetic.cli.profiles import resolve_infra
from kinetic.collections_helpers import (
  append_child_to_manifest,
  build_initial_manifest,
  call_with_input,
)
from kinetic.constants import build_bucket_name
from kinetic.job_status import JobStatus
from kinetic.jobs import _TERMINAL_STATUSES, JobHandle
from kinetic.utils import storage

_DEFAULT_MAX_CONCURRENT = 64
_STATUS_POLL_INTERVAL = 5.0
_MANIFEST_POLL_INTERVAL = 10.0

# Upper bound on how long `attach_batch()` waits for a still-submitting
# `map()` to name its remaining children in the manifest.  A bound is
# required: without one a stalled or dead submitter leaves `wait()` and
# `results()` blocked forever.  Pass `poll_timeout=None` to opt out.
_DEFAULT_MANIFEST_POLL_TIMEOUT = 1800.0


def _resolve_bucket(
  project: str | None, cluster: str | None
) -> tuple[str, str]:
  """Return `(resolved_project, bucket_name)`.

  Resolution follows the standard chain: explicit kwarg > KINETIC_* env var
  > active profile field > built-in default.
  """
  infra = resolve_infra(project=project, cluster=cluster)
  return infra["project"], build_bucket_name(infra["project"], infra["cluster"])


class BatchError(Exception):
  """Raised when a batch collection has failed children.

  Attributes:
    group_id: The collection's group identifier.
    failures: List of JobHandles for children that were submitted and
      then failed.  Never contains `None`, so `job.job_id` and
      `job.status()` are always safe to call on its items.
    partial_results: List where successful positions contain the
      result and failed positions contain `None`.
    submission_failures: Mapping of input index to the exception raised
      while submitting that input.  Those inputs never became jobs, so
      they have no `JobHandle` and never appear in `failures`.
  """

  def __init__(
    self,
    group_id: str,
    failures: list[JobHandle],
    partial_results: list[Any],
    submission_failures: dict[int, Exception] | None = None,
  ):
    self.group_id = group_id
    self.failures = failures
    self.partial_results = partial_results
    self.submission_failures = dict(submission_failures or {})
    n_failed = len(failures) + len(self.submission_failures)
    n_total = len(partial_results)
    super().__init__(f"Batch {group_id}: {n_failed} of {n_total} jobs failed")


@dataclass
class BatchHandle:
  """Handle for a collection of submitted jobs.

  Created by `run_async_map()` or reconstructed by
  `kinetic.attach_batch()`.  Provides collection-level observation,
  result gathering, and cleanup.
  """

  group_id: str
  name: str | None
  tags: dict[str, str]
  jobs: list[JobHandle | None]

  # Bucket / project derived from eager resolution in map().
  _bucket_name: str = field(default="", repr=False, compare=False)
  _project: str = field(default="", repr=False, compare=False)

  # Internal state for background submission.
  _submission_complete: threading.Event = field(
    default_factory=threading.Event, repr=False, compare=False
  )
  _submission_error: BaseException | None = field(
    default=None, repr=False, compare=False
  )
  _lock: threading.Lock = field(
    default_factory=threading.Lock, repr=False, compare=False
  )

  # Per-index submission errors (index -> exception).
  _submission_errors: dict[int, Exception] = field(
    default_factory=dict, repr=False, compare=False
  )

  # Cached failure list populated by results() so that failures()
  # remains accurate after cleanup deletes K8s resources.
  _cached_failures: list[JobHandle] | None = field(
    default=None, repr=False, compare=False
  )

  # Set by cancel().  The submission loop reads both: the event stops it
  # launching queued inputs, and the index set stops it reading the
  # resulting NOT_FOUND statuses as failures worth retrying.
  _cancel_requested: threading.Event = field(
    default_factory=threading.Event, repr=False, compare=False
  )
  _cancelled_indices: set[int] = field(
    default_factory=set, repr=False, compare=False
  )

  # Children the manifest names but whose handle.json could not be
  # downloaded (index -> job_id).  Populated by attach_batch().
  _unavailable_children: dict[int, str] = field(
    default_factory=dict, repr=False, compare=False
  )

  def statuses(self) -> list[tuple[int, JobStatus]]:
    """Return `(index, status)` for each submitted job."""
    return [
      (i, job.status()) for i, job in enumerate(self.jobs) if job is not None
    ]

  def status_counts(self) -> dict[str, int]:
    """Return a count of jobs in each status."""
    return dict(collections.Counter(s.value for _, s in self.statuses()))

  def _all_accounted_for(self, seen: set[int]) -> bool:
    """True when no further job can reach a terminal state.

    Submission being complete freezes the `jobs` list: a slot still
    holding `None` will never hold a job.  Its input raised at
    submission time, `cancel()` stopped it launching, or `attach_batch()`
    could not load its handle.  Waiting on those slots would never end,
    so what is left to wait for is every job that does exist reaching a
    terminal state.
    """
    if not self._submission_complete.is_set():
      return False
    with self._lock:
      submitted = {i for i, job in enumerate(self.jobs) if job is not None}
    return submitted <= seen

  def wait(self, *, timeout: float | None = None) -> None:
    """Block until all jobs reach a terminal state."""
    deadline = None if timeout is None else time.monotonic() + timeout

    # Wait for background submission to finish first.
    if not self._submission_complete.is_set():
      remaining = (
        None if deadline is None else max(0, deadline - time.monotonic())
      )
      if not self._submission_complete.wait(timeout=remaining):
        raise TimeoutError(
          f"Timed out waiting for submission to complete "
          f"for batch {self.group_id}"
        )

    if self._submission_error is not None:
      raise self._submission_error

    # Poll until every submitted job is terminal.
    while True:
      if all(
        job.status() in _TERMINAL_STATUSES
        for job in self.jobs
        if job is not None
      ):
        break
      if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(
          f"Timed out waiting for batch {self.group_id} after {timeout}s"
        )
      time.sleep(_STATUS_POLL_INTERVAL)

    if self._submission_errors:
      logging.warning(
        "Batch %s: %d input(s) failed at submission time. "
        "Use handle.submission_failures to inspect.",
        self.group_id,
        len(self._submission_errors),
      )

  def as_completed(
    self,
    *,
    poll_interval: float = 5.0,
    timeout: float | None = None,
  ) -> Iterator[JobHandle]:
    """Yield jobs as they reach terminal states, in completion order.

    Unlike the simple approach of waiting for all submissions first,
    this streams results as soon as each job reaches a terminal state
    — even while more inputs are still being submitted.

    Args:
      poll_interval: Seconds between status polls.
      timeout: Maximum seconds to wait.  Raises `TimeoutError` if
        exceeded.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    seen: set[int] = set()

    while True:
      # Snapshot current jobs (slots may be filled by the submission thread).
      with self._lock:
        current_jobs = list(enumerate(self.jobs))

      newly_done = []
      for i, job in current_jobs:
        if i in seen or job is None:
          continue
        if job.status() in _TERMINAL_STATUSES:
          newly_done.append(i)

      for i in newly_done:
        seen.add(i)
        yield self.jobs[i]  # type: ignore[misc]

      if self._all_accounted_for(seen):
        break

      if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(
          f"as_completed() timed out after {timeout}s for batch {self.group_id}"
        )

      if not newly_done:
        time.sleep(poll_interval)

  def results(
    self,
    *,
    timeout: float | None = None,
    ordered: bool = True,
    cleanup: bool = True,
    return_exceptions: bool = False,
  ) -> list[Any]:
    """Collect results from all jobs.

    Args:
      timeout: Maximum seconds to wait for all jobs.
      ordered: If *True*, return in input order.  If *False*,
        return in completion order.
      cleanup: If *True*, clean up each child's K8s and GCS
        resources (the group manifest is preserved). Note that
        cleaning up causes `failures()` to return an empty list
        as job statuses become `NOT_FOUND`.
      return_exceptions: If *True*, failed positions contain the
        exception object.  If *False*, raise `BatchError` on any
        failure — including inputs that failed at submission time,
        which are reported by `BatchError.submission_failures`.

    Returns:
      List of results (input order when *ordered=True*, completion
      order otherwise).
    """
    unavailable = self.unavailable_children
    if unavailable:
      logging.warning(
        "Batch %s: %d child(ren) at indices %s have no handle in GCS, so "
        "their results cannot be collected; those positions are None. "
        "Their artifacts were most likely already deleted by an earlier "
        "results(cleanup=True) or cleanup() call.",
        self.group_id,
        len(unavailable),
        sorted(unavailable),
      )

    if ordered:
      results_list, failures = self._results_ordered(
        timeout=timeout, cleanup=cleanup, return_exceptions=return_exceptions
      )
    else:
      results_list, failures = self._results_completion_order(
        timeout=timeout, cleanup=cleanup, return_exceptions=return_exceptions
      )

    submission_failures = self.submission_failures
    if (failures or submission_failures) and not return_exceptions:
      raise BatchError(
        group_id=self.group_id,
        failures=failures,
        partial_results=results_list,
        submission_failures=submission_failures,
      )

    return results_list

  def _results_ordered(
    self,
    *,
    timeout: float | None,
    cleanup: bool,
    return_exceptions: bool,
  ) -> tuple[list[Any], list[JobHandle]]:
    """Collect results in input order (waits for all jobs first).

    The returned failure list holds only real `JobHandle` objects.
    Inputs that never became jobs are reported separately through
    `submission_failures`.
    """
    self.wait(timeout=timeout)
    failures: list[JobHandle] = []
    results_list: list[Any] = [None] * len(self.jobs)

    for i, job in enumerate(self.jobs):
      if job is None:
        if return_exceptions and i in self._submission_errors:
          results_list[i] = self._submission_errors[i]
        continue
      try:
        results_list[i] = job.result(cleanup=cleanup)
      except Exception as exc:
        if return_exceptions:
          results_list[i] = exc
        failures.append(job)

    with self._lock:
      self._cached_failures = list(failures)
    return results_list, failures

  def _results_completion_order(
    self,
    *,
    timeout: float | None,
    cleanup: bool,
    return_exceptions: bool,
  ) -> tuple[list[Any], list[JobHandle]]:
    """Collect results in completion order, streaming as they arrive."""
    failures: list[JobHandle] = []
    results_list: list[Any] = []

    for job in self.as_completed(timeout=timeout):
      try:
        results_list.append(job.result(cleanup=cleanup))
      except Exception as exc:
        if return_exceptions:
          results_list.append(exc)
        failures.append(job)

    if return_exceptions:
      for idx in sorted(self._submission_errors):
        results_list.append(self._submission_errors[idx])

    with self._lock:
      self._cached_failures = list(failures)
    return results_list, failures

  def failures(self) -> list[JobHandle]:
    """Return handles for jobs that failed.

    Only includes jobs whose status is `FAILED`.  Jobs that are
    `NOT_FOUND` (e.g. after cleanup) are excluded because the
    status is ambiguous — use `statuses()` for finer control.

    After `results()` has been called, this returns the cached
    failure list from that collection pass, so it remains accurate
    even if cleanup has deleted K8s resources.

    See Also:
      `submission_failures`: returns per-input errors for inputs
      that failed at submission time (`jobs[idx]` is `None`).
    """
    with self._lock:
      if self._cached_failures is not None:
        return list(self._cached_failures)
    return [
      job
      for job in self.jobs
      if job is not None and job.status() == JobStatus.FAILED
    ]

  @property
  def submission_failures(self) -> dict[int, Exception]:
    """Return a copy of per-input submission errors (index -> exception).

    These are inputs where the submission itself failed (e.g. validation
    error, network error).  The corresponding `jobs[idx]` slot is
    `None`.  These errors are included in `results()` output but are
    **not** reflected by `failures()` which only inspects live job
    statuses.
    """
    with self._lock:
      return dict(self._submission_errors)

  @property
  def unavailable_children(self) -> dict[int, str]:
    """Return an index-to-job-id map of children whose handle is missing.

    Only `attach_batch()` fills this in.  Each entry is a child that the
    group manifest names, and that was therefore submitted, whose
    `handle.json` could not be downloaded.  Its GCS prefix is almost
    always gone because an earlier `results(cleanup=True)` or
    `cleanup()` deleted it, so its `jobs[idx]` slot stays `None` and its
    result can never be collected again.

    A `None` slot that this mapping does not name is a different case:
    an input that the original `map()` never submitted.
    """
    with self._lock:
      return dict(self._unavailable_children)

  def cancel(self) -> None:
    """Cancel the collection: stop launching, then stop what is running.

    Cancellation covers the whole collection, not just the jobs that
    happen to be live right now.  Queued inputs that a bounded
    `max_concurrent` has not launched yet are dropped, and the cancelled
    children are marked so that a batch running with `retries > 0` does
    not read their `NOT_FOUND` status as a failure and resubmit them.

    Already-terminal children are left alone.  Cancelling deletes each
    child's Kubernetes resource and keeps its GCS artifacts.
    """
    self._cancel_requested.set()

    with self._lock:
      # Every slot is off-limits from here on: the submitted ones are
      # cancelled below, and the rest must never launch.
      self._cancelled_indices.update(range(len(self.jobs)))
      snapshot = list(self.jobs)

    for job in snapshot:
      if job is None:
        continue
      try:
        if job.status() not in _TERMINAL_STATUSES:
          job.cancel()
      except RuntimeError:
        logging.warning("Failed to cancel job %s", job.job_id)

  def cleanup(self, *, k8s: bool = True, gcs: bool = True) -> None:
    """Clean up all jobs and optionally the group manifest.

    Args:
      k8s: Delete K8s resources for each child.
      gcs: Delete GCS artifacts for each child **and** the group
        manifest.
    """
    for job in self.jobs:
      if job is None:
        continue
      try:
        job.cleanup(k8s=k8s, gcs=gcs)
      except (RuntimeError, google_exceptions.GoogleAPIError):
        logging.warning("Failed to clean up job %s", job.job_id)

    if gcs:
      bucket = self._bucket_name
      project = self._project
      if not bucket and self.jobs:
        first = next((j for j in self.jobs if j is not None), None)
        if first is not None:
          bucket = first.bucket_name
          project = first.project
      if bucket:
        try:
          storage.cleanup_manifest(bucket, self.group_id, project=project)
        except google_exceptions.GoogleAPIError:
          logging.warning(
            "Failed to clean up manifest for group %s", self.group_id
          )


def _child_index(child: dict, total_expected: int) -> int | None:
  """Return a manifest child's `group_index`, or `None` if unusable."""
  idx = child.get("group_index")
  if not isinstance(idx, int) or idx < 0 or idx >= total_expected:
    return None
  return idx


def _load_child_handle(
  bucket_name: str,
  child: dict,
  project: str,
) -> JobHandle | None:
  """Download and reconstruct a single child handle.

  Returns `None` when the handle cannot be read back, which for a child
  the manifest already names means its GCS prefix is gone rather than
  not written yet.
  """
  try:
    payload = storage.download_handle(
      bucket_name, child["job_id"], project=project
    )
    return JobHandle.from_dict(payload)
  except (google_exceptions.GoogleAPIError, KeyError, ValueError):
    logging.warning(
      "Could not load handle for child job %s; skipping",
      child.get("job_id"),
    )
    return None


def _listed_indices(manifest: dict, total_expected: int) -> set[int]:
  """Return the child indices the manifest already claims.

  A claimed index is one the original `map()` has submitted, whether or
  not its `handle.json` can still be downloaded.  This is what tells
  "not submitted yet" (worth waiting for) apart from "submitted, handle
  gone" (never going to load, e.g. after `results(cleanup=True)` deleted
  the child's GCS prefix).
  """
  return {
    idx
    for child in manifest.get("children", [])
    if (idx := _child_index(child, total_expected)) is not None
  }


def _hydrate_children(
  handle: BatchHandle,
  manifest: dict,
  bucket_name: str,
  project: str,
  total_expected: int,
) -> None:
  """Fill *handle*'s empty slots from the manifest's children.

  A named child whose `handle.json` cannot be downloaded is recorded in
  `handle._unavailable_children` instead of being left indistinguishable
  from one that was never submitted.  Slots that already hold a handle
  are left alone, so this is safe to call repeatedly as the manifest
  grows.
  """
  for child in manifest.get("children", []):
    idx = _child_index(child, total_expected)
    if idx is None:
      logging.warning(
        "Invalid child index %r (total_expected=%d); skipping",
        child.get("group_index"),
        total_expected,
      )
      continue

    with handle._lock:
      if handle.jobs[idx] is not None:
        continue

    job_handle = _load_child_handle(bucket_name, child, project)
    with handle._lock:
      if job_handle is None:
        handle._unavailable_children[idx] = child.get("job_id", "<unknown>")
      else:
        handle.jobs[idx] = job_handle
        handle._unavailable_children.pop(idx, None)


def _manifest_poll_loop(
  handle: BatchHandle,
  bucket_name: str,
  group_id: str,
  project: str,
  total_expected: int,
  poll_interval: float,
  timeout: float | None,
) -> None:
  """Poll GCS manifest until all children appear, then set `_submission_complete`.

  Used by `attach_batch()` when the manifest names fewer children than
  `total_expected`, indicating the original `map()` is still submitting.

  The loop stops once the manifest *names* every child, not once every
  handle loads.  A named child whose handle cannot be downloaded is
  already as resolved as it will ever get, so waiting on it would only
  burn the timeout and leave `wait()` blocked in the meantime.
  """
  deadline = None if timeout is None else time.monotonic() + timeout

  try:
    while True:
      if deadline is not None and time.monotonic() >= deadline:
        logging.warning(
          "Timed out polling manifest for batch %s (%d/%d children). "
          "Reattach again to pick up any children submitted since.",
          group_id,
          sum(1 for j in handle.jobs if j is not None),
          total_expected,
        )
        break

      time.sleep(poll_interval)

      try:
        manifest = storage.download_manifest(
          bucket_name, group_id, project=project
        )
      except google_exceptions.GoogleAPIError:
        logging.warning("Failed to poll manifest for batch %s", group_id)
        continue

      _hydrate_children(handle, manifest, bucket_name, project, total_expected)

      if len(_listed_indices(manifest, total_expected)) >= total_expected:
        break
  finally:
    handle._submission_complete.set()


def _cancel_active(handle: BatchHandle, active_indices: set[int]) -> None:
  """Best-effort cancel of all active jobs."""
  for idx in list(active_indices):
    job = handle.jobs[idx]
    if job is None:
      continue
    try:
      job.cancel()
    except RuntimeError:
      logging.warning("Failed to cancel job at index %d", idx)


@dataclass
class _SubmissionState:
  """Groups the mutable state tracked by the submission loop.

  Provides named predicates so the main loop reads as a clear
  sequence of phases rather than a tangle of flags and counters.
  """

  handle: BatchHandle
  manifest: dict
  submit_fn: Any
  inputs: list
  input_mode: str
  max_concurrent: int | None
  max_attempts: int
  fail_fast: bool
  cancel_running_on_fail: bool

  attempt_counts: list[int] = field(init=False)
  pending: collections.deque = field(init=False)
  active: set[int] = field(default_factory=set, init=False)
  stop_launching: bool = field(default=False, init=False)

  def __post_init__(self):
    self.attempt_counts = [0] * len(self.inputs)
    self.pending = collections.deque(range(len(self.inputs)))

  @property
  def has_work(self) -> bool:
    """True while jobs remain to be submitted or are still running."""
    return bool(self.pending) or bool(self.active)

  @property
  def cancelled(self) -> bool:
    """True once `BatchHandle.cancel()` has been called."""
    return self.handle._cancel_requested.is_set()

  def can_submit_more(self) -> bool:
    """True when the next pending job is allowed to launch."""
    if not self.pending or self.stop_launching or self.cancelled:
      return False
    return self.max_concurrent is None or len(self.active) < self.max_concurrent

  def needs_active_polling(self) -> bool:
    """True when the loop must poll active jobs itself.

    Polling only earns its keep when a terminal status would change what
    the loop does next: launch a queued input, retry a failed attempt, or
    cancel the running siblings.  When none of those apply the caller
    observes terminal states through `wait()` / `results()` instead, so
    the loop exits and stops holding up whoever called `map()`.

    Note that `fail_fast` alone does not require polling: with nothing
    left to launch and no siblings to cancel, a failure has no effect the
    loop could act on.
    """
    if not self.active:
      return False
    if self.pending or self.max_attempts > 1:
      return True
    return self.fail_fast and self.cancel_running_on_fail

  def trigger_fail_fast(self) -> None:
    """Stop launching new jobs and optionally cancel siblings."""
    self.stop_launching = True
    if self.cancel_running_on_fail:
      _cancel_active(self.handle, self.active)


def _submit_available(state: _SubmissionState) -> None:
  """Submit pending jobs up to the concurrency limit.

  On per-input errors the exception is recorded in
  `handle._submission_errors` and, when `fail_fast` is set,
  `trigger_fail_fast` is called.
  """
  handle = state.handle
  launched: list[int] = []

  while state.can_submit_more():
    idx = state.pending.popleft()
    state.attempt_counts[idx] += 1

    # attempt submission
    try:
      job_handle = call_with_input(
        state.submit_fn, state.inputs[idx], state.input_mode
      )
    except Exception as exc:
      logging.error("Submission failed for index %d: %s", idx, exc)
      with handle._lock:
        handle._submission_errors[idx] = exc
      if state.fail_fast:
        state.trigger_fail_fast()
      continue

    # tag with group metadata and persist
    job_handle.group_id = handle.group_id
    job_handle.group_kind = state.manifest["group_kind"]
    job_handle.group_index = idx

    try:
      storage.upload_handle(
        job_handle.bucket_name,
        job_handle.job_id,
        job_handle.to_dict(),
        project=job_handle.project,
      )
    except google_exceptions.GoogleAPIError:
      logging.warning(
        "Failed to re-upload handle with group fields for %s",
        job_handle.job_id,
      )

    # register in handle and manifest
    with handle._lock:
      handle.jobs[idx] = job_handle
    state.active.add(idx)
    launched.append(idx)

    append_child_to_manifest(
      state.manifest, idx, job_handle.job_id, state.attempt_counts[idx]
    )
    try:
      storage.upload_manifest(
        handle._bucket_name,
        handle.group_id,
        state.manifest,
        project=handle._project,
      )
    except google_exceptions.GoogleAPIError:
      logging.warning(
        "Failed to update manifest after submitting index %d", idx
      )

  if state.stop_launching or state.cancelled:
    state.pending.clear()

  if state.cancelled and launched:
    # `cancel()` sets its flag before it snapshots `handle.jobs`, so any
    # job registered after that snapshot is one `cancel()` could not see.
    # Cancelling those here is what closes the window between the two.
    _cancel_active(handle, set(launched))


def _poll_and_handle_terminal(state: _SubmissionState) -> None:
  """Poll active jobs for terminal states; retry or trigger fail_fast."""
  handle = state.handle

  # Collect all newly-terminal jobs in one pass.
  newly_terminal: list[tuple[int, JobStatus, JobHandle]] = []
  for idx in list(state.active):
    job = handle.jobs[idx]
    if job is None:
      continue
    try:
      status = job.status()
      if status in _TERMINAL_STATUSES:
        newly_terminal.append((idx, status, job))
    except (RuntimeError, google_exceptions.GoogleAPIError):
      logging.warning("Failed to poll status for index %d", idx)

  with handle._lock:
    cancelled_indices = set(handle._cancelled_indices)

  for idx, status, job in newly_terminal:
    state.active.discard(idx)

    if status not in (JobStatus.FAILED, JobStatus.NOT_FOUND):
      continue

    if idx in cancelled_indices:
      # Cancelling deletes the child's K8s resource, so its status turns
      # NOT_FOUND.  That is the requested outcome — resubmitting it would
      # undo the cancellation, and failing the batch over it would report
      # a failure the caller asked for.
      continue

    if state.attempt_counts[idx] < state.max_attempts:
      # Retry: clean up previous attempt's K8s resources and re-queue.
      try:
        job.cleanup(k8s=True, gcs=False)
      except RuntimeError:
        logging.warning("Failed to clean up before retry for index %d", idx)
      state.pending.append(idx)
    elif state.fail_fast:
      state.trigger_fail_fast()


def _runs_in_calling_thread(
  max_concurrent: int | None,
  retries: int,
  fail_fast: bool,
  cancel_running_on_fail: bool,
) -> bool:
  """True when the submission loop can finish without outliving `map()`.

  That needs every input to launch on the first pass (no concurrency
  limit) *and* nothing that would keep the loop polling afterwards — no
  retries to schedule, and no fail-fast cancellation to perform.  Any
  other combination has to run in a background thread, or `map()` would
  block until the whole batch finishes instead of returning a handle.

  Mirrors `_SubmissionState.needs_active_polling`; keep the two in step.
  """
  if max_concurrent is not None or retries > 0:
    return False
  return not (fail_fast and cancel_running_on_fail)


def _submission_loop(
  submit_fn,
  inputs: list,
  input_mode: str,
  manifest: dict,
  handle: BatchHandle,
  max_concurrent: int | None,
  retries: int,
  fail_fast: bool,
  cancel_running_on_fail: bool,
) -> None:
  """Core submission and retry loop.

  Mutates *handle.jobs* and *manifest* in place.  Runs in the calling
  thread or in a background thread, as decided by
  `_runs_in_calling_thread`.

  Each iteration follows three phases:

  1. **Submit** — launch pending jobs up to the concurrency limit.
  2. **Poll**  — check active jobs for terminal states, retry or
     trigger `fail_fast` as needed.
  3. **Sleep** — back off before the next poll cycle.
  """
  state = _SubmissionState(
    handle=handle,
    manifest=manifest,
    submit_fn=submit_fn,
    inputs=inputs,
    input_mode=input_mode,
    max_concurrent=max_concurrent,
    max_attempts=1 + retries,
    fail_fast=fail_fast,
    cancel_running_on_fail=cancel_running_on_fail,
  )

  try:
    while state.has_work:
      _submit_available(state)

      if not state.needs_active_polling():
        break

      _poll_and_handle_terminal(state)

      if state.has_work:
        time.sleep(_STATUS_POLL_INTERVAL)

  except BaseException as exc:
    handle._submission_error = exc
    logging.error("Submission loop error: %s", exc)
  finally:
    handle._submission_complete.set()


def map(
  submit_fn,
  inputs,
  *,
  input_mode: str = "auto",
  max_concurrent: int | None = _DEFAULT_MAX_CONCURRENT,
  retries: int = 0,
  fail_fast: bool = False,
  cancel_running_on_fail: bool = False,
  name: str | None = None,
  tags: dict[str, str] | None = None,
  project: str | None = None,
  cluster: str | None = None,
) -> BatchHandle:
  """Launch many independent jobs over a set of inputs.

  `submit_fn` must be a function obtained from `func.run_async` where
  `func` is decorated with `@kinetic.run(...)`. Each input is dispatched according to
  `input_mode` and submitted as a separate remote job.

  Args:
    submit_fn: A callable obtained from `func.run_async`.
    inputs: Iterable of inputs to fan out over.
    input_mode: How each input item is passed to *submit_fn*.
      `"auto"` (default) dispatches dicts as `**kwargs`,
      lists/tuples as `*args`, and scalars as a single positional
      argument.
    max_concurrent: Maximum number of concurrently active jobs.
      `None` submits all immediately.
    retries: Number of additional attempts after a job failure.
    fail_fast: Stop launching new jobs after the first failure.
    cancel_running_on_fail: Cancel running siblings on failure.
    name: Human-readable collection name.
    tags: Arbitrary key-value metadata.
    project: GCP project. Falls back to KINETIC_PROJECT, then the active
      profile's project, then GOOGLE_CLOUD_PROJECT.
    cluster: GKE cluster name. Falls back to KINETIC_CLUSTER, then the
      active profile's cluster, then the built-in default.

  Returns:
    A `BatchHandle` for observing, collecting, and cleaning up
    the collection.  Returns as soon as submission is either finished
    in the calling thread or handed to a background thread — never
    after the jobs themselves finish.  Use `wait()` or `results()` to
    block on the batch.
  """
  if not callable(submit_fn):
    raise TypeError("submit_fn must be callable")

  if max_concurrent is not None and max_concurrent < 1:
    raise ValueError(
      f"max_concurrent must be a positive integer, got {max_concurrent}"
    )

  if retries < 0:
    raise ValueError(f"retries must be non-negative, got {retries}")

  if input_mode not in ("auto", "single", "args", "kwargs"):
    raise ValueError(f"Unknown input_mode: {input_mode!r}")

  inputs = list(inputs)
  if not inputs:
    raise ValueError("inputs must be non-empty")

  # Resolve bucket eagerly so the initial manifest can be written
  # before any jobs are submitted.
  resolved_project, bucket_name = _resolve_bucket(project, cluster)

  group_id = f"grp-{uuid.uuid4().hex[:8]}"
  group_kind = "map"
  fn_name = getattr(submit_fn, "__name__", str(submit_fn))

  manifest = build_initial_manifest(
    group_id, group_kind, name, tags, len(inputs), fn_name
  )

  # Write the initial manifest (empty children) before any jobs are
  # submitted so that crash recovery can distinguish "0 of N
  # submitted" from "collection never created".
  storage.upload_manifest(
    bucket_name, group_id, manifest, project=resolved_project
  )

  # Pre-allocate the jobs list with None placeholders.
  jobs: list[JobHandle | None] = [None] * len(inputs)

  handle = BatchHandle(
    group_id=group_id,
    name=name,
    tags=tags or {},
    jobs=jobs,
    _bucket_name=bucket_name,
    _project=resolved_project,
  )

  if max_concurrent is None and len(inputs) > 100:
    logging.warning(
      "Submitting %d jobs with max_concurrent=None. "
      "Consider setting max_concurrent to limit resource usage.",
      len(inputs),
    )

  if _runs_in_calling_thread(
    max_concurrent, retries, fail_fast, cancel_running_on_fail
  ):
    # Simple path: submit all in calling thread.
    _submission_loop(
      submit_fn=submit_fn,
      inputs=inputs,
      input_mode=input_mode,
      manifest=manifest,
      handle=handle,
      max_concurrent=max_concurrent,
      retries=retries,
      fail_fast=fail_fast,
      cancel_running_on_fail=cancel_running_on_fail,
    )
  else:
    # Background thread for bounded concurrency or retries.
    thread = threading.Thread(
      target=_submission_loop,
      kwargs={
        "submit_fn": submit_fn,
        "inputs": inputs,
        "input_mode": input_mode,
        "manifest": manifest,
        "handle": handle,
        "max_concurrent": max_concurrent,
        "retries": retries,
        "fail_fast": fail_fast,
        "cancel_running_on_fail": cancel_running_on_fail,
      },
      daemon=False,
    )
    thread.start()

  return handle


def attach_batch(
  group_id: str,
  project: str | None = None,
  cluster: str | None = None,
  poll_interval: float = _MANIFEST_POLL_INTERVAL,
  poll_timeout: float | None = _DEFAULT_MANIFEST_POLL_TIMEOUT,
) -> BatchHandle:
  """Reattach to an existing batch collection by *group_id*.

  Downloads the group manifest from GCS, reconstructs `JobHandle`
  objects for each child, and returns a fully usable `BatchHandle`.

  If the manifest names fewer children than `total_expected` (i.e. the
  original `map()` is still submitting), the returned handle polls the
  manifest in a background thread until the rest are named or
  *poll_timeout* is reached.

  A child the manifest names but whose `handle.json` cannot be
  downloaded is *not* treated as still-pending — its GCS artifacts have
  typically been cleaned up (by `results(cleanup=True)` or
  `JobHandle.cleanup()`), so no amount of polling will produce it.  Its
  slot stays `None` and the batch is reported as fully submitted, which
  keeps `wait()` and `results()` from blocking on a job that no longer
  exists.

  Args:
    group_id: The collection identifier (e.g. `"grp-a1b2c3d4"`).
    project: GCP project. Falls back to KINETIC_PROJECT, then the active
      profile's project, then GOOGLE_CLOUD_PROJECT.
    cluster: GKE cluster name. Falls back to KINETIC_CLUSTER, then the
      active profile's cluster, then the built-in default.
    poll_interval: Seconds between manifest polls when the batch
      is partially submitted.
    poll_timeout: Maximum seconds to poll for remaining children,
      30 minutes by default.  On timeout the handle reports submission
      as complete and the missing slots stay `None`; reattach again to
      pick up children submitted since.  `None` polls indefinitely and
      risks blocking `wait()` forever if the submitter has died.

  Returns:
    A hydrated `BatchHandle` ready for `results()`, etc.
  """
  resolved_project, bucket_name = _resolve_bucket(project, cluster)

  manifest = storage.download_manifest(
    bucket_name, group_id, project=resolved_project
  )

  children = manifest.get("children", [])
  total_expected = manifest.get("total_expected", len(children))

  # Preallocate to total_expected and slot each child by group_index
  # so that index alignment is preserved even when some handles are
  # missing or the batch was only partially submitted.
  handle = BatchHandle(
    group_id=manifest["group_id"],
    name=manifest.get("group_name"),
    tags=manifest.get("tags", {}),
    jobs=[None] * total_expected,
    _bucket_name=bucket_name,
    _project=resolved_project,
  )

  _hydrate_children(
    handle, manifest, bucket_name, resolved_project, total_expected
  )

  unavailable = handle.unavailable_children
  if unavailable:
    logging.warning(
      "Batch %s: %d of %d children have no handle in GCS (indices %s). "
      "Their artifacts were already deleted — usually by an earlier "
      "results(cleanup=True) — so their results cannot be collected "
      "again and their slots stay None.",
      group_id,
      len(unavailable),
      total_expected,
      sorted(unavailable),
    )

  # Completeness is decided by what the manifest *names*, not by how
  # many handles loaded.  A named child whose handle is gone will never
  # load, so polling for it would only stall wait() and results().
  listed = _listed_indices(manifest, total_expected)
  if len(listed) >= total_expected:
    handle._submission_complete.set()
  else:
    logging.warning(
      "Batch %s was partially submitted: %d of %d expected jobs are "
      "recorded in the manifest. Polling for the remaining children.",
      group_id,
      len(listed),
      total_expected,
    )
    thread = threading.Thread(
      target=_manifest_poll_loop,
      kwargs={
        "handle": handle,
        "bucket_name": bucket_name,
        "group_id": group_id,
        "project": resolved_project,
        "total_expected": total_expected,
        "poll_interval": poll_interval,
        "timeout": poll_timeout,
      },
      daemon=True,
    )
    thread.start()

  return handle
