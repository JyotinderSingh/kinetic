# Interactive Debugging

This page explains how to attach a debugger to a job that runs on the
cluster. Pass `debug=True` to `@kinetic.run()`. The pod then starts a
`debugpy` server and waits for your editor before it calls your
function. You set breakpoints, step through the function, and inspect
variables on the accelerator that runs the code. Read this page when a
job fails in a way that the logs do not explain, or when you want to
explore data on the pod.

Kinetic prints a `launch.json` entry that uses the standard `debugpy`
attach request. VS Code, and other editors that use the VS Code Python
debugger, for example Cursor and VSCodium, can use that entry without changes.

## Before you start

- An active profile. `kinetic init` creates the profile. Kinetic reads
  the project and the cluster from the profile, so the commands on this
  page do not need `--project` or `--cluster`.
- `kubectl` on your `PATH`. Kinetic runs `kubectl port-forward` to
  connect your editor to the pod.
- VS Code, or another editor, with the Python and Python Debugger
  extensions.

## A first debug session

Add `debug=True` to `@kinetic.run()`. Then call the function directly:

```python
import kinetic


@kinetic.run(accelerator="tpu-v5litepod-4", debug=True)
def train():
  import jax

  breakpoint()  # the debugger pauses here
  x = jax.numpy.arange(16)
  return x.sum()


train()
```

When you call `train()`, Kinetic does these things:

:::{container} kinetic-steps
1. **Submits the job.** Kinetic packages the function and creates the
   job on the cluster with debug mode enabled.
2. **Waits for the pod.** The pod installs `debugpy`, starts a `debugpy`
   server on port 5678, and writes a ready marker to Cloud Storage. Kinetic
   waits for that marker and prints the job status while it waits.
3. **Opens a tunnel.** Kinetic starts `kubectl port-forward` from
   `localhost:5678` to the pod.
4. **Prints a `launch.json` entry.** Paste the entry into
   `.vscode/launch.json` in your workspace.
:::

Then, in your editor, press **F5** (Run > Start Debugging). The debugger
attaches and pauses inside the Kinetic runner, one line before the call
to your function. Press **F11** to step into your function, or press
**F10** to step over the call. In the second case, the debugger pauses
at the first `breakpoint()` inside your function.

The blocking call returns the return value of your function when the
function ends. Kinetic then stops the port-forward process.

:::{tip}
You do not need to call `breakpoint()`. Set breakpoints in the editor as
you do for a local run. The debugger attaches before your function
starts, so those breakpoints are active from the first line. If the
editor marks a breakpoint as unverified, see
[Path mappings and source files](#path-mappings-and-source-files).
:::

:::{note}
A blocking debug call does not stream the pod log to your terminal. To
read the log during the session, run `kinetic jobs logs <job_id> --follow`
in a second terminal.
:::

## What `debug=True` changes

`debug=True` changes the pod and the local call in these ways:

- The pod runs `uv pip install --system debugpy` before it starts the
  `debugpy` server. The image that Kinetic builds contains `uv`. If you
  use your own image, `uv` must be on the `PATH`. See
  [Container Images](containers.md).
- The pod sets `PYTHONBREAKPOINT=debugpy.breakpoint`, so a `breakpoint()`
  call in your code pauses in the attached debugger.
- The pod waits up to 10 minutes for a debugger to attach. If no debugger
  connects in that window, the pod runs your function without a debugger.
  A `breakpoint()` call does not pause the function when no debugger is
  attached.
- Kinetic does not stream the pod log during a blocking debug call.
- Kinetic does not delete the job resources when the function ends. See
  [Clean up after a debug session](#clean-up-after-a-debug-session).

## Attach later from the command line

For a long session, or when you want to attach from a different machine,
submit the job as a detached job with `run_async()`. A blocking call blocks
and returns the value of the function, not a `JobHandle`:

```python
import kinetic


@kinetic.run(accelerator="tpu-v5litepod-4", debug=True)
def train():
  import jax

  breakpoint()
  ...


job = train.run_async()
print(job.job_id)  # for example: job-a1b2c3d4
```

`run_async()` returns at once. The pod starts, and then waits up to 10
minutes for a debugger. Attach from a terminal on the same machine, or on
a different machine with the same active profile:

```bash
kinetic jobs debug <job_id>
```

`kinetic jobs debug` waits for the pod, opens the tunnel, and prints the
`launch.json` entry. The command blocks until the job ends or until you
press Ctrl+C. The command then stops the port-forward process. The
command fails at once if you did not submit the job with `debug=True`.

You can do the same from Python:

```python
import kinetic
from kinetic.debug import cleanup_port_forward

job = kinetic.attach("<job_id>")
pf = job.debug_attach(local_port=5678)
try:
  value = job.result()
finally:
  cleanup_port_forward(pf)
job.cleanup()
```

`job.result()` waits for the job to end and returns the value of the
function. For a debug job, `result()` does not delete the job resources.
Call `job.cleanup()` when you no longer need the job resources.

## Port conflicts

The default local port is `5678`. That port is the `debugpy` default, and
the VS Code Python extension fills it in for an attach configuration. If
another process listens on `5678` on your machine, `kubectl port-forward`
exits at once and Kinetic raises a `RuntimeError`. Attach again on a
different port:

```bash
kinetic jobs debug <job_id> --port 5679
```

Or from Python:

```python
pf = job.debug_attach(local_port=5679)
```

Change the `port` field in the printed `launch.json` entry to the same
value.

:::{note}
A blocking call always uses port `5678`. If the port is in use, the call
raises `RuntimeError` after Kinetic submits the job. The job stays on the
cluster and waits for a debugger. Find the job ID in the log output or
with `kinetic jobs list`. Then attach to the job with
`kinetic jobs debug <job_id> --port <port>`.
:::

## Path mappings and source files

The debugger must match the source files in your editor with the source
files on the pod. Kinetic does two things:

- The runner creates a symbolic link on the pod at the absolute path of
  your **entry directory**, the directory of the file that defines your
  function. The link points at the root of the extracted source. The
  runner creates the link only if that path does not exist on the pod.
  The source files in your entry directory therefore exist on the pod
  under the same absolute path as on your machine. When the entry
  directory sits below the package root, the link still points at the
  root of the extracted source. See
  [What Ships to the Pod](packaging.md#the-package-root).
- The printed `launch.json` entry contains a `pathMappings` entry. The
  `localRoot` value is the entry directory for a blocking call, and
  `${workspaceFolder}` for `kinetic jobs debug`. The `working_dir=`
  argument of `debug_attach()` sets that value. The `remoteRoot` value is
  `/tmp/workspace`. The runner does not use that directory. The runner
  extracts the source into a temporary directory with the prefix
  `kinetic-run-`.

If the editor marks a breakpoint as unverified, do one of these things:

- Delete the `pathMappings` entry from the configuration. Then start the
  debugger again. The symbolic link makes your local paths valid on the
  pod, so the debugger does not need a mapping.
- Put a `breakpoint()` call in the function. That call always pauses
  when a debugger is attached.

## The attach window

Two waits have a fixed limit of 10 minutes each. Kinetic has no option
to change these limits.

- **On the pod.** After the `debugpy` server is ready, the pod waits up
  to 10 minutes for a debugger. If no debugger connects, the pod runs the
  function without a debugger. The job does not wait forever.
- **On your machine.** `debug_attach()`, and therefore a blocking call
  and `kinetic jobs debug`, wait up to 10 minutes for the ready marker.
  This wait includes the time to schedule the pod and to install
  `debugpy`. On a node pool that is scaled to zero, the wait can end with
  a `TimeoutError` while the job continues. In that case, run
  `kinetic jobs debug <job_id>`. The command waits again for the pod.

The 2-hour value that applies to debug jobs is not the attach window. That
value is the retention time of the finished Kubernetes Job. See the next
section.

## Clean up after a debug session

A blocking debug call, and `result()` on a debug job, use
`cleanup=False`. Kinetic keeps the job resources so that you can inspect
them after the session:

- **The Kubernetes resource** stays on the cluster after the pod ends. On
  the GKE backend, Kubernetes deletes the finished Job 2 hours after it
  ends (10 minutes for a normal job). On the Pathways backend, Kinetic
  sets no automatic deletion for the LeaderWorkerSet.
- **The artifacts in the jobs bucket** (`context.zip`, `payload.pkl`,
  `result.pkl`, `handle.json`) stay until you delete them, or until the
  30-day rule of the bucket deletes them.

Delete both when you are done:

```bash
kinetic jobs cleanup <job_id>
```

Or from Python, call `job.cleanup()`. To collect the result and delete
the resources in one step, call `job.result(cleanup=True)`. The command
`kinetic jobs result <job_id>` does the same by default.

## Multi-host debugging

On a multi-host TPU slice, for example `tpu-v5litepod-16`, Kinetic uses
the Pathways backend and starts one pod per host. Only the leader pod
runs the `debugpy` server. You attach one time, to the leader. The worker
pods wait for a marker that the leader writes to Cloud Storage after the
debugger attaches, or after the attach window ends. The workers therefore
do not start the distributed runtime while the leader waits for a
debugger.

The workers wait up to 11 minutes for the marker (the 10-minute attach
window plus a 1-minute margin). If the marker does not appear in that
time, for example because the leader pod failed, the workers fail with a
`RuntimeError` that names the missing marker.

See [Distributed Training](distributed_training.md) for the multi-host
slice names.

:::{warning}
Do not use `spot=True` with `debug=True`. If Google Cloud preempts the
node during the session, the pod stops and the debug connection drops.
Kinetic warns at decoration time when both are set. Use on-demand
capacity for interactive work.
:::

## Automated environments

A blocking call with `debug=True` needs an interactive terminal. If
`stdin` is not a TTY (CI, `nohup`, piped input), the call raises
`RuntimeError`. Kinetic runs this check after it submits the job. The
submitted job therefore stays on the cluster: the pod waits 10 minutes
for a debugger and then runs the function.

For automation, use `run_async()`. The detached call does not check the
terminal. Attach later with `kinetic jobs debug <job_id>` from an
interactive shell, or let the job run without a debugger.

To permit a blocking debug call without a TTY, set
`KINETIC_NO_TTY_DEBUG=1` in the environment of the process. See
[Configuration](../configuration.md).

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em` Detached Jobs
:link: async_jobs
:link-type: doc

`run_async()`, `JobHandle`, and the `kinetic jobs` commands that pair with
`kinetic jobs debug`.
:::

:::{grid-item-card} {octicon}`server;1em` Distributed Training
:link: distributed_training
:link-type: doc

Multi-host TPU slices and the Pathways backend.
:::

:::{grid-item-card} {octicon}`gear;1em` Configuration
:link: ../configuration
:link-type: doc

`KINETIC_NO_TTY_DEBUG` and the other settings that Kinetic reads.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

What to check when a pod stays in `PENDING`, or when a job fails before
it calls your function.
:::
::::
