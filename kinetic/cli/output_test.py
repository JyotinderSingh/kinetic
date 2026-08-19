"""Tests for kinetic.cli.output — LiveOutputPanel and state rendering."""

import io
from unittest import mock

from absl.testing import absltest, parameterized
from rich.console import Console
from rich.text import Text

from kinetic.cli import output
from kinetic.cli.output import LiveOutputPanel


def _make_non_terminal_console():
  """Create a Console that reports is_terminal=False."""
  return Console(force_terminal=False, file=open("/dev/null", "w"))


class MakePanelTest(absltest.TestCase):
  """Tests for _make_panel rendering logic."""

  def test_shows_last_max_lines(self):
    panel = LiveOutputPanel("Title", max_lines=3)
    for i in range(10):
      panel._lines.append(f"line {i}")

    content = panel._make_panel().renderable

    self.assertNotIn("line 6", content)
    self.assertIn("line 7", content)
    self.assertIn("line 9", content)

  def test_error_shows_all_lines(self):
    panel = LiveOutputPanel("Title", max_lines=3)
    for i in range(10):
      panel._lines.append(f"line {i}")
    panel._has_error = True

    content = panel._make_panel().renderable

    self.assertIn("line 0", content)
    self.assertIn("line 9", content)

  def test_subtitle_suppressed_on_error(self):
    panel = LiveOutputPanel("Title")
    panel._has_error = True

    self.assertIsNone(panel._make_panel().subtitle)

  def test_subtitle_suppressed_when_show_subtitle_false(self):
    panel = LiveOutputPanel("Title", show_subtitle=False)
    panel._start_time = 0
    panel._phrase_order = list(range(10))

    self.assertIsNone(panel._make_panel().subtitle)


class TransientBehaviorTest(absltest.TestCase):
  """Tests for transient panel clearing/persistence on exit."""

  def test_transient_clears_on_success(self):
    panel = LiveOutputPanel("Test", transient=True)
    panel._live = mock.MagicMock()
    panel.on_output("some output")

    panel.__exit__(None, None, None)

    update_calls = [
      c
      for c in panel._live.update.call_args_list
      if isinstance(c.args[0], Text)
    ]
    self.assertLen(update_calls, 1)

  def test_transient_persists_on_mark_error(self):
    panel = LiveOutputPanel("Test", transient=True)
    panel._live = mock.MagicMock()
    panel.on_output("some output")
    panel.mark_error()

    panel.__exit__(None, None, None)

    update_calls = [
      c
      for c in panel._live.update.call_args_list
      if isinstance(c.args[0], Text)
    ]
    self.assertEmpty(update_calls)

  def test_transient_persists_on_exception(self):
    console = _make_non_terminal_console()
    panel = LiveOutputPanel("Test", transient=True, target_console=console)

    with self.assertRaises(RuntimeError), panel:
      raise RuntimeError("fail")

    self.assertTrue(panel._has_error)

  def test_exception_sets_has_error_without_mark_error(self):
    console = _make_non_terminal_console()
    panel = LiveOutputPanel("Test", target_console=console)

    with self.assertRaises(TypeError), panel:
      raise TypeError("bad")

    self.assertTrue(panel._has_error)


def _gpu_export(**overrides):
  """A GPU entry as written by program._export_stack_outputs."""
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


def _render(outputs):
  """Render infrastructure_state to plain text.

  The table is wide so values are not truncated mid-word.
  """
  console = Console(file=io.StringIO(), width=200, record=True)
  with mock.patch.object(output, "console", console):
    output.infrastructure_state(
      {key: mock.MagicMock(value=value) for key, value in outputs.items()}
    )
  return console.export_text()


class InfrastructureStateAcceleratorTest(parameterized.TestCase):
  """`pool list` and `status` must show how a pool was provisioned.

  Spot and reservation change what a pool costs and whether it can be
  preempted, so they belong in the state table alongside machine type.
  """

  @parameterized.named_parameters(
    dict(testcase_name="spot", spot=True, expected="Spot"),
    dict(testcase_name="on_demand", spot=False, expected="On-demand"),
  )
  def test_shows_provisioning_model(self, spot, expected):
    text = _render({"accelerators": [_gpu_export(spot=spot)]})

    self.assertIn("Provisioning", text)
    self.assertIn(expected, text)

  def test_shows_reservation(self):
    text = _render({"accelerators": [_gpu_export(reservation="my-res")]})

    self.assertIn("Reservation", text)
    self.assertIn("my-res", text)

  def test_omits_reservation_row_when_unset(self):
    text = _render({"accelerators": [_gpu_export()]})

    self.assertNotIn("Reservation", text)

  def test_omits_unknown_fields_from_legacy_export(self):
    """Pre-export stacks record neither field, so claim neither."""
    entry = _gpu_export()
    legacy = {
      k: v for k, v in entry.items() if k not in ("spot", "reservation")
    }

    text = _render({"accelerators": [legacy]})

    self.assertNotIn("Provisioning", text)
    self.assertNotIn("Reservation", text)

  def test_each_pool_shows_its_own_settings(self):
    text = _render(
      {
        "accelerators": [
          _gpu_export(node_pool="gpu-a100-abcd", spot=True),
          _gpu_export(node_pool="gpu-a100-ef01", reservation="my-res"),
        ]
      }
    )

    self.assertIn("Spot", text)
    self.assertIn("On-demand", text)
    self.assertIn("my-res", text)

  def test_legacy_single_accelerator_output(self):
    text = _render({"accelerator": _gpu_export(spot=True)})

    self.assertIn("Provisioning", text)
    self.assertIn("Spot", text)

  def test_no_pools(self):
    text = _render({"accelerators": []})

    self.assertIn("CPU only", text)


if __name__ == "__main__":
  absltest.main()
