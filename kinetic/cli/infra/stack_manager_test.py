"""Tests for kinetic.cli.infra.stack_manager."""

from unittest import mock

from absl.testing import absltest, parameterized

from kinetic.cli.config import InfraConfig
from kinetic.cli.infra import stack_manager
from kinetic.core import accelerators


class GetStackTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    # Stub out the Pulumi automation API so the test only exercises the
    # backend wiring.
    self.mock_pulumi_cmd = self.enterContext(
      mock.patch.object(stack_manager.auto, "PulumiCommand")
    )
    self.mock_project_settings = self.enterContext(
      mock.patch.object(stack_manager.auto, "ProjectSettings")
    )
    self.mock_project_backend = self.enterContext(
      mock.patch.object(stack_manager.auto, "ProjectBackend")
    )
    self.mock_create_or_select_stack = self.enterContext(
      mock.patch.object(stack_manager.auto, "create_or_select_stack")
    )
    self.mock_local_workspace = self.enterContext(
      mock.patch.object(stack_manager.auto, "LocalWorkspaceOptions")
    )
    self.mock_ensure_gcs = self.enterContext(
      mock.patch.object(stack_manager, "ensure_gcs_backend")
    )

  def test_uses_per_project_gcs_bucket(self):
    config = InfraConfig(project="kinetic-proj", zone="z", cluster_name="c")

    stack_manager.get_stack(lambda: None, config)

    self.assertEqual(
      self.mock_project_backend.call_args.kwargs["url"],
      "gs://kinetic-proj-kinetic-state",
    )

  def test_ensures_bucket_with_project(self):
    config = InfraConfig(project="kinetic-proj", zone="z", cluster_name="c")

    stack_manager.get_stack(lambda: None, config)

    self.mock_ensure_gcs.assert_called_once_with("kinetic-proj")


def _gpu_export(**overrides):
  """A GPU entry as written by _export_stack_outputs."""
  return {
    "type": "GPU",
    "name": "a100",
    "count": 1,
    "machine_type": "a2-highgpu-1g",
    "node_pool": "gpu-a100-abcd",
    "node_count": 1,
    "min_nodes": 0,
    "spot": False,
    "reservation": None,
  } | overrides


def _tpu_export(**overrides):
  """A TPU entry as written by _export_stack_outputs."""
  return {
    "type": "TPU",
    "name": "v6e",
    "chips": 8,
    "topology": "2x4",
    "machine_type": "ct6e-standard-4t",
    "node_pool": "tpu-v6e-abcd",
    "node_count": 2,
    "min_nodes": 0,
    "spot": False,
    "reservation": None,
  } | overrides


class ExportToNodePoolTest(parameterized.TestCase):
  """Stack exports are the only record of how a pool was provisioned.

  Whatever this drops is re-applied as a default on the next update,
  replacing the pool without the setting.
  """

  @parameterized.named_parameters(
    dict(testcase_name="gpu_spot", entry=_gpu_export(spot=True), spot=True),
    dict(testcase_name="gpu_on_demand", entry=_gpu_export(), spot=False),
    dict(testcase_name="tpu_spot", entry=_tpu_export(spot=True), spot=True),
    dict(testcase_name="tpu_on_demand", entry=_tpu_export(), spot=False),
  )
  def test_reads_spot(self, entry, spot):
    pool = stack_manager._export_to_node_pool(entry)

    self.assertEqual(pool.accelerator.spot, spot)

  @parameterized.named_parameters(
    dict(testcase_name="gpu", entry=_gpu_export(reservation="my-res")),
    dict(testcase_name="tpu", entry=_tpu_export(reservation="my-res")),
  )
  def test_reads_reservation(self, entry):
    pool = stack_manager._export_to_node_pool(entry)

    self.assertEqual(pool.reservation, "my-res")

  @parameterized.named_parameters(
    dict(testcase_name="gpu", entry=_gpu_export()),
    dict(testcase_name="tpu", entry=_tpu_export()),
  )
  def test_absent_reservation_is_none(self, entry):
    pool = stack_manager._export_to_node_pool(entry)

    self.assertIsNone(pool.reservation)

  @parameterized.named_parameters(
    dict(testcase_name="gpu", entry=_gpu_export()),
    dict(testcase_name="tpu", entry=_tpu_export()),
  )
  def test_legacy_export_without_spot_or_reservation(self, entry):
    """Stacks last updated before these keys were exported still load."""
    legacy = {
      k: v for k, v in entry.items() if k not in ("spot", "reservation")
    }

    pool = stack_manager._export_to_node_pool(legacy)

    self.assertFalse(pool.accelerator.spot)
    self.assertIsNone(pool.reservation)

  def test_preserves_accelerator_shape_and_min_nodes(self):
    pool = stack_manager._export_to_node_pool(
      _tpu_export(min_nodes=2, spot=True)
    )

    self.assertEqual(pool.name, "tpu-v6e-abcd")
    self.assertEqual(pool.min_nodes, 2)
    self.assertEqual(
      pool.accelerator, accelerators.make_tpu("v6e", 8, spot=True)
    )

  def test_unknown_type_raises(self):
    with self.assertRaises(ValueError):
      stack_manager._export_to_node_pool(_gpu_export(type="QPU"))


class GetCurrentNodePoolsTest(absltest.TestCase):
  def _stack_with_outputs(self, outputs):
    stack = mock.MagicMock()
    stack.outputs.return_value = {
      key: mock.MagicMock(value=value) for key, value in outputs.items()
    }
    return stack

  def test_reads_spot_and_reservation_from_accelerators(self):
    stack = self._stack_with_outputs(
      {
        "accelerators": [
          _gpu_export(spot=True),
          _tpu_export(reservation="my-res"),
        ]
      }
    )

    pools = stack_manager.get_current_node_pools(stack)

    self.assertLen(pools, 2)
    self.assertTrue(pools[0].accelerator.spot)
    self.assertEqual(pools[1].reservation, "my-res")

  def test_legacy_single_accelerator_output(self):
    stack = self._stack_with_outputs({"accelerator": _gpu_export(spot=True)})

    pools = stack_manager.get_current_node_pools(stack)

    self.assertLen(pools, 1)
    self.assertTrue(pools[0].accelerator.spot)

  def test_no_accelerator_outputs(self):
    stack = self._stack_with_outputs({"project": "p"})

    self.assertEmpty(stack_manager.get_current_node_pools(stack))


if __name__ == "__main__":
  absltest.main()
