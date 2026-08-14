"""Tests for kinetic.utils.storage — pure-local helpers.

Everything that touches Cloud Storage is tested for real against the
fake-gcs-server emulator in tests/integration/storage_contract_test.py;
only the helpers with no GCS involvement live here.
"""

import os
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest, parameterized

from kinetic.constants import get_default_project
from kinetic.utils.storage import _compute_total_size


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


class TestGetProject(parameterized.TestCase):
  @parameterized.named_parameters(
    dict(
      testcase_name="kinetic_project_only",
      kn_project="kn-proj",
      gc_project=None,
      expected="kn-proj",
    ),
    dict(
      testcase_name="google_cloud_project_fallback",
      kn_project=None,
      gc_project="gc-proj",
      expected="gc-proj",
    ),
    dict(
      testcase_name="neither_set",
      kn_project=None,
      gc_project=None,
      expected=None,
    ),
    dict(
      testcase_name="kinetic_takes_precedence",
      kn_project="kn-proj",
      gc_project="gc-proj",
      expected="kn-proj",
    ),
  )
  def test_resolves_project(self, kn_project, gc_project, expected):
    env = {}
    if kn_project:
      env["KINETIC_PROJECT"] = kn_project
    if gc_project:
      env["GOOGLE_CLOUD_PROJECT"] = gc_project
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(get_default_project(), expected)


class TestComputeTotalSize(absltest.TestCase):
  def test_single_file(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.bin"
    f.write_bytes(b"x" * 100)
    self.assertEqual(_compute_total_size(str(f)), 100)

  def test_directory(self):
    tmp = _make_temp_path(self)
    d = tmp / "dir"
    d.mkdir()
    (d / "a.txt").write_bytes(b"x" * 50)
    (d / "b.txt").write_bytes(b"y" * 30)
    self.assertEqual(_compute_total_size(str(d)), 80)

  def test_empty_directory(self):
    tmp = _make_temp_path(self)
    d = tmp / "empty"
    d.mkdir()
    self.assertEqual(_compute_total_size(str(d)), 0)


if __name__ == "__main__":
  absltest.main()
