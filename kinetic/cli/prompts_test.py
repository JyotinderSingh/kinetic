"""Tests for kinetic.cli.prompts."""

import json
from unittest import mock

from absl.testing import absltest

from kinetic.cli import prompts


class TestPrompts(absltest.TestCase):
  def _assert_delimited(self, args, *positionals):
    """Assert each positional follows ``--`` and no flag does.

    gcloud stops parsing options at the ``--`` delimiter, so a flag placed
    after it is read as a positional and the command fails with
    "unrecognized arguments".
    """
    self.assertIn("--", args)
    idx_delim = args.index("--")
    for positional in positionals:
      self.assertIn(positional, args)
      self.assertGreater(args.index(positional), idx_delim)
    trailing_flags = [a for a in args[idx_delim + 1 :] if a.startswith("-")]
    self.assertEmpty(
      trailing_flags,
      f"{trailing_flags} follow the -- delimiter, so gcloud parses them as "
      "positional arguments rather than flags",
    )

  @mock.patch("kinetic.cli.prompts.subprocess.run")
  def test_project_exists_args(self, mock_run):
    mock_run.return_value.returncode = 0
    prompts._project_exists("my-proj")
    mock_run.assert_called_once()
    self._assert_delimited(mock_run.call_args[0][0], "my-proj")

  @mock.patch("kinetic.cli.prompts.subprocess.run")
  def test_create_project_args(self, mock_run):
    mock_run.return_value.returncode = 0
    prompts._create_project("my-proj")
    mock_run.assert_called_once()
    self._assert_delimited(mock_run.call_args[0][0], "my-proj")

  @mock.patch("kinetic.cli.prompts.subprocess.run")
  @mock.patch("click.confirm", return_value=True)
  def test_link_billing_account_args(self, mock_confirm, mock_run):
    # Mock first call (list billing accounts)
    mock_acct = {
      "name": "billingAccounts/123",
      "displayName": "My Billing Account",
    }
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = json.dumps([mock_acct])

    # _link_billing_account calls subprocess.run twice (list and link).

    prompts._link_billing_account("my-proj")

    self.assertEqual(mock_run.call_count, 2)

    # Check second call args
    args = mock_run.call_args_list[1][0][0]
    self._assert_delimited(args, "my-proj")
    # --billing-account is required, so it must land before the delimiter
    # where gcloud still parses it as a flag.
    self.assertIn("--billing-account=123", args)
    self.assertLess(args.index("--billing-account=123"), args.index("--"))


if __name__ == "__main__":
  absltest.main()
