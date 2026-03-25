"""kinetic jobs command group — inspect and manage async jobs."""

import click
from rich.table import Table

from kinetic.cli.options import jobs_options
from kinetic.cli.output import banner, console, success, warning
from kinetic.jobs import attach, list_jobs


def _attach(job_id, project, cluster_name, zone, namespace):
  """Attach to a job, validating required options."""
  if not project:
    raise click.UsageError(
      "Project is required. Set --project or KINETIC_PROJECT."
    )
  return attach(job_id, project=project, cluster=cluster_name)


@click.group()
def jobs():
  """Inspect and manage async remote jobs."""


@jobs.command("list")
@jobs_options
def list_command(project, zone, cluster_name, namespace):
  """List live async jobs."""
  if not project:
    raise click.UsageError(
      "Project is required. Set --project or KINETIC_PROJECT."
    )

  banner("kinetic Jobs")

  handles = list_jobs(
    project=project,
    zone=zone,
    cluster=cluster_name,
    namespace=namespace,
  )
  if not handles:
    warning("No live jobs found.")
    return

  table = Table(title="Async Jobs")
  table.add_column("Job ID", style="bold")
  table.add_column("Function", style="green")
  table.add_column("Accelerator")
  table.add_column("Backend")
  table.add_column("Created", style="dim")

  for handle in handles:
    table.add_row(
      handle.job_id,
      handle.func_name,
      handle.accelerator,
      handle.backend,
      handle.created_at,
    )

  console.print()
  console.print(table)


@jobs.command()
@click.argument("job_id")
@jobs_options
def status(job_id, project, zone, cluster_name, namespace):
  """Show the current status for a job."""
  handle = _attach(job_id, project, cluster_name, zone, namespace)
  console.print(f"{job_id}: {handle.status().value}")


@jobs.command()
@click.argument("job_id")
@click.option("--follow", "-f", is_flag=True, help="Stream logs until completion.")
@click.option(
  "--tail", "-n",
  type=int,
  default=None,
  help="Show the last N log lines instead of the full log.",
)
@jobs_options
def logs(job_id, follow, tail, project, zone, cluster_name, namespace):
  """Show or stream logs for a job."""
  if follow and tail is not None:
    raise click.ClickException("Use either --follow or --tail, not both.")

  handle = _attach(job_id, project, cluster_name, zone, namespace)

  if follow:
    handle.logs(follow=True)
    return

  if tail is not None:
    console.print(handle.tail(n=tail))
    return

  text = handle.logs()
  if text:
    console.print(text)


@jobs.command()
@click.argument("job_id")
@click.option(
  "--timeout",
  type=float,
  default=None,
  help="Maximum seconds to wait for the result.",
)
@click.option(
  "--cleanup/--no-cleanup",
  default=True,
  help="Delete k8s and GCS artifacts after collecting the result.",
)
@jobs_options
def result(job_id, timeout, cleanup, project, zone, cluster_name, namespace):
  """Wait for and print a job result."""
  handle = _attach(job_id, project, cluster_name, zone, namespace)
  console.print(handle.result(timeout=timeout, cleanup=cleanup))


@jobs.command()
@click.argument("job_id")
@jobs_options
def cancel(job_id, project, zone, cluster_name, namespace):
  """Cancel a running job by deleting its k8s resource."""
  handle = _attach(job_id, project, cluster_name, zone, namespace)
  handle.cancel()
  success(f"Cancelled {job_id}")


@jobs.command()
@click.argument("job_id")
@click.option(
  "--k8s/--no-k8s",
  default=True,
  help="Delete Kubernetes resources.",
)
@click.option(
  "--gcs/--no-gcs",
  default=True,
  help="Delete uploaded GCS artifacts.",
)
@jobs_options
def cleanup(job_id, k8s, gcs, project, zone, cluster_name, namespace):
  """Clean up Kubernetes and/or GCS resources for a job."""
  handle = _attach(job_id, project, cluster_name, zone, namespace)
  handle.cleanup(k8s=k8s, gcs=gcs)
  success(f"Cleaned up {job_id}")
