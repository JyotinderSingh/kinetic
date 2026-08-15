"""Tests for kinetic.cli.infra.program — node pool and K8s resources."""

from unittest import mock

from absl.testing import absltest, parameterized

from kinetic.cli.config import NodePoolConfig
from kinetic.cli.infra import stack_manager
from kinetic.core.accelerators import GpuConfig, TpuConfig

# Patch pulumi provider modules before importing program, so the module-level
# imports inside program.py pick up the mocks.
with mock.patch.dict(
  "sys.modules",
  {
    "pulumi_command": mock.MagicMock(),
    "pulumi_gcp": mock.MagicMock(),
    "pulumi_kubernetes": mock.MagicMock(),
  },
):
  from kinetic.cli.infra import program


class TestCreateTpuNodePool(parameterized.TestCase):
  """Verify _create_tpu_node_pool sets placement_policy correctly.

  Multi-host TPUs require COMPACT placement with an explicit topology;
  single-host slices must NOT have placement_policy or GKE rejects
  the node pool.
  """

  @parameterized.named_parameters(
    dict(
      testcase_name="v5p_multi_host",
      tpu=TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2),
      expect_placement=True,
    ),
    dict(
      testcase_name="v6e_multi_host",
      tpu=TpuConfig("v6e", 8, "2x4", "tpu-v6e-slice", "ct6e-standard-4t", 2),
      expect_placement=True,
    ),
    dict(
      testcase_name="v3_single_host",
      tpu=TpuConfig("v3", 4, "2x2", "tpu-v3-podslice", "ct3-hightpu-4t", 1),
      expect_placement=False,
    ),
    dict(
      testcase_name="v5litepod_single_host",
      tpu=TpuConfig(
        "v5litepod", 4, "2x2", "tpu-v5-lite-podslice", "ct5lp-hightpu-4t", 1
      ),
      expect_placement=False,
    ),
  )
  @mock.patch.object(program, "gcp")
  def test_placement_policy(self, gcp_mock, tpu, expect_placement):
    cluster = mock.MagicMock()
    cluster.name = "test-cluster"

    program._create_tpu_node_pool(
      cluster,
      tpu,
      "us-central2-b",
      "my-project",
      f"tpu-{tpu.name}-abcd",
      "sa@test.iam.gserviceaccount.com",
    )

    call_kwargs = gcp_mock.container.NodePool.call_args
    placement = call_kwargs.kwargs.get(
      "placement_policy", call_kwargs[1].get("placement_policy")
    )

    if expect_placement:
      self.assertIsNotNone(placement)
      gcp_mock.container.NodePoolPlacementPolicyArgs.assert_called_once_with(
        type="COMPACT",
        tpu_topology=tpu.topology,
      )
    else:
      self.assertIsNone(placement)


def _make_config(node_pools=None, force_destroy=True):
  """Create a mock InfraConfig for testing."""
  config = mock.MagicMock()
  config.project = "test-project"
  config.zone = "us-central1-a"
  config.cluster_name = "test-cluster"
  config.node_pools = node_pools or []
  config.force_destroy = force_destroy
  return config


class TestGpuDriverConditional(absltest.TestCase):
  """GPU driver DaemonSet must only be installed when GPU pools are present."""

  def _run_program(self, config=None):
    config = config or _make_config()
    with (
      mock.patch.object(program, "pulumi"),
      mock.patch.object(program, "command"),
      mock.patch.object(program, "gcp"),
      mock.patch.object(program, "k8s") as k8s_mock,
    ):
      program.create_program(config)()
    return k8s_mock

  def test_installed_when_gpu_pools_present(self):
    gpu = GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
    config = _make_config([NodePoolConfig("gpu-l4-a3f2", gpu)])
    k8s_mock = self._run_program(config)

    gpu_calls = [
      c
      for c in k8s_mock.yaml.ConfigFile.call_args_list
      if c.args[0] == "nvidia-gpu-drivers"
    ]
    self.assertLen(gpu_calls, 1)

  def test_not_installed_for_cpu_only(self):
    k8s_mock = self._run_program(_make_config([]))

    gpu_calls = [
      c
      for c in k8s_mock.yaml.ConfigFile.call_args_list
      if c.args[0] == "nvidia-gpu-drivers"
    ]
    self.assertEmpty(gpu_calls)

  def test_not_installed_for_tpu_only(self):
    tpu = TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2)
    config = _make_config([NodePoolConfig("tpu-v5p-b7e1", tpu)])
    k8s_mock = self._run_program(config)

    gpu_calls = [
      c
      for c in k8s_mock.yaml.ConfigFile.call_args_list
      if c.args[0] == "nvidia-gpu-drivers"
    ]
    self.assertEmpty(gpu_calls)


class TestForceDestroy(parameterized.TestCase):
  """force_destroy on InfraConfig must flow to both GCS buckets."""

  @parameterized.named_parameters(
    dict(testcase_name="enabled", force_destroy=True),
    dict(testcase_name="disabled", force_destroy=False),
  )
  def test_buckets_receive_force_destroy(self, force_destroy):
    config = _make_config(force_destroy=force_destroy)

    with (
      mock.patch.object(program, "pulumi"),
      mock.patch.object(program, "command"),
      mock.patch.object(program, "gcp") as gcp_mock,
      mock.patch.object(program, "k8s"),
    ):
      program.create_program(config)()

    bucket_calls = gcp_mock.storage.Bucket.call_args_list
    self.assertLen(bucket_calls, 2)
    for call in bucket_calls:
      self.assertEqual(call.kwargs["force_destroy"], force_destroy)

  def test_force_destroy_is_exported(self):
    config = _make_config(force_destroy=False)

    with (
      mock.patch.object(program, "pulumi") as pulumi_mock,
      mock.patch.object(program, "command"),
      mock.patch.object(program, "gcp"),
      mock.patch.object(program, "k8s"),
    ):
      program.create_program(config)()

    exported = {
      call.args[0]: call.args[1] for call in pulumi_mock.export.call_args_list
    }
    self.assertIn("force_destroy", exported)
    self.assertFalse(exported["force_destroy"])


_SPOT_GPU = GpuConfig(
  "a100", 1, "nvidia-tesla-a100", "a2-highgpu-1g", spot=True
)
_SPOT_TPU = TpuConfig(
  "v6e", 8, "2x4", "tpu-v6e-slice", "ct6e-standard-4t", 2, spot=True
)
_ON_DEMAND_GPU = GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
_ON_DEMAND_TPU = TpuConfig(
  "v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2
)


class _ResolvedOutputs(list):
  """Stands in for ``pulumi.Output.all`` — the resolved values, as a list.

  ``apply`` stays lazy (it returns a mock without running the callback),
  matching how the fully mocked ``pulumi`` behaves elsewhere in this
  file. ``_build_kubeconfig`` calls it with mock cluster attributes that
  its ``json.dumps`` cannot serialize.
  """

  def apply(self, fn):
    del fn
    return mock.MagicMock()


def _run_program_capturing_exports(config):
  """Run the Pulumi program, returning (stack exports, gcp mock).

  ``NodePool.name.apply`` is stubbed to evaluate eagerly, and
  ``Output.all`` to return the resolved list, so the exported
  accelerator entries are plain dicts rather than Pulumi Outputs.
  """
  exports = {}

  with (
    mock.patch.object(program, "pulumi") as pulumi_mock,
    mock.patch.object(program, "command"),
    mock.patch.object(program, "gcp") as gcp_mock,
    mock.patch.object(program, "k8s"),
  ):
    pulumi_mock.export.side_effect = exports.__setitem__
    pulumi_mock.Output.all.side_effect = lambda *entries: _ResolvedOutputs(
      entries
    )

    def make_node_pool(_resource_name, **kwargs):
      pool = mock.MagicMock()
      pool.name.apply.side_effect = lambda fn: fn(kwargs["name"])
      return pool

    gcp_mock.container.NodePool.side_effect = make_node_pool
    program.create_program(config)()

  return exports, gcp_mock


class TestAcceleratorExportRoundTrip(parameterized.TestCase):
  """Pool settings must survive export → load_state → re-apply.

  ``pool add``/``pool remove`` rebuild the whole pool list from the
  ``accelerators`` stack export and re-declare every existing pool. A
  setting the export omits comes back as its default, and because GKE
  node_config changes force replacement, the pool is silently rebuilt
  without it.
  """

  @parameterized.named_parameters(
    dict(testcase_name="gpu_spot", accel=_SPOT_GPU, min_nodes=1),
    dict(testcase_name="tpu_spot", accel=_SPOT_TPU, min_nodes=2),
    dict(testcase_name="gpu_on_demand", accel=_ON_DEMAND_GPU, min_nodes=0),
    dict(testcase_name="tpu_on_demand", accel=_ON_DEMAND_TPU, min_nodes=0),
  )
  def test_spot_survives_round_trip(self, accel, min_nodes):
    pool = NodePoolConfig("pool-abcd", accel, min_nodes=min_nodes)
    exports, _ = _run_program_capturing_exports(_make_config([pool]))

    (entry,) = exports["accelerators"]
    self.assertEqual(entry["spot"], accel.spot)
    self.assertEqual(stack_manager._export_to_node_pool(entry), pool)

  @parameterized.named_parameters(
    dict(testcase_name="gpu", accel=_ON_DEMAND_GPU),
    dict(testcase_name="tpu", accel=_ON_DEMAND_TPU),
  )
  def test_reservation_survives_round_trip(self, accel):
    pool = NodePoolConfig("pool-abcd", accel, reservation="my-reservation")
    exports, _ = _run_program_capturing_exports(_make_config([pool]))

    (entry,) = exports["accelerators"]
    self.assertEqual(entry["reservation"], "my-reservation")
    self.assertEqual(stack_manager._export_to_node_pool(entry), pool)

  def test_multiple_pools_keep_their_own_settings(self):
    pools = [
      NodePoolConfig("gpu-a100-abcd", _SPOT_GPU),
      NodePoolConfig("gpu-l4-ef01", _ON_DEMAND_GPU, reservation="l4-res"),
    ]
    exports, _ = _run_program_capturing_exports(_make_config(pools))

    restored = [
      stack_manager._export_to_node_pool(e) for e in exports["accelerators"]
    ]
    self.assertEqual(restored, pools)

  @parameterized.named_parameters(
    dict(testcase_name="spot", accel=_SPOT_GPU, reservation=None),
    dict(
      testcase_name="reservation", accel=_ON_DEMAND_GPU, reservation="my-res"
    ),
  )
  def test_reapplying_exported_state_keeps_node_config(
    self, accel, reservation
  ):
    """The regression: a second `pool add` must not reset the first pool.

    Simulates `pool add --spot` followed by another pool command —
    export, read back via load_state, re-declare — and checks the node
    config Pulumi would apply the second time still carries the
    setting.
    """
    original = NodePoolConfig("pool-abcd", accel, reservation=reservation)
    exports, _ = _run_program_capturing_exports(_make_config([original]))

    restored = [
      stack_manager._export_to_node_pool(e) for e in exports["accelerators"]
    ]
    new_pool = NodePoolConfig("gpu-l4-new1", _ON_DEMAND_GPU)
    _, gcp_mock = _run_program_capturing_exports(
      _make_config(restored + [new_pool])
    )

    node_config = gcp_mock.container.NodePoolNodeConfigArgs.call_args_list[0]
    self.assertEqual(node_config.kwargs["spot"], accel.spot)
    if reservation is None:
      self.assertIsNone(node_config.kwargs["reservation_affinity"])
    else:
      gcp_mock.container.NodePoolNodeConfigReservationAffinityArgs.assert_called_once_with(
        consume_reservation_type="SPECIFIC_RESERVATION",
        key="compute.googleapis.com/reservation-name",
        values=[reservation],
      )


class TestClusterResourceLabels(absltest.TestCase):
  """The GKE cluster must carry a kinetic resource label.

  GCP resource labels are how operators identify which clusters in a
  project were provisioned by kinetic versus by other tools.
  """

  def test_cluster_has_kinetic_resource_label(self):
    config = _make_config()

    with (
      mock.patch.object(program, "pulumi"),
      mock.patch.object(program, "command"),
      mock.patch.object(program, "gcp") as gcp_mock,
      mock.patch.object(program, "k8s"),
    ):
      program.create_program(config)()

    cluster_call = gcp_mock.container.Cluster.call_args
    self.assertIsNotNone(cluster_call)
    self.assertEqual(
      cluster_call.kwargs["resource_labels"],
      {
        program.RESOURCE_NAME_PREFIX: "true",
        "goog-packaged-solution": "kinetic",
      },
    )

  def test_default_node_pool_has_kinetic_label(self):
    """The default GKE node pool must carry the same label as accelerator pools."""
    config = _make_config()

    with (
      mock.patch.object(program, "pulumi"),
      mock.patch.object(program, "command"),
      mock.patch.object(program, "gcp") as gcp_mock,
      mock.patch.object(program, "k8s"),
    ):
      program.create_program(config)()

    cluster_call = gcp_mock.container.Cluster.call_args
    node_config_call = gcp_mock.container.ClusterNodeConfigArgs.call_args_list[
      -1
    ]
    self.assertIsNotNone(cluster_call)
    self.assertEqual(
      node_config_call.kwargs["labels"],
      {
        program.RESOURCE_NAME_PREFIX: "true",
        "goog-packaged-solution": "kinetic",
      },
    )


if __name__ == "__main__":
  absltest.main()
