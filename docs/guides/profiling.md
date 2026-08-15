# Performance Profiling

A job that runs on a TPU or a GPU does not always use the accelerator
well. A performance profile (a trace) shows where the accelerator time
goes. This page covers
**XProf**, the profiler for XLA workloads. It shows how to capture a
trace inside a Kinetic job and how to keep the trace after the pod ends.
It also shows how to view the trace on your own machine.

The pod filesystem is gone when your function returns. The workflow
therefore has three parts:

1. Capture the trace inside the decorated function.
2. Write the trace under `KINETIC_OUTPUT_DIR`, which is a Cloud Storage
   location that stays after the pod ends.
3. View the trace on your machine.

## What XProf is

XProf is the open-source accelerator profiler from
[OpenXLA](https://openxla.org/xprof). It was formerly the TensorBoard
"profile" plugin. XProf runs as a standalone tool or as a TensorBoard
tab. It reads accelerator hardware counters, so the capture window adds
little overhead to the job.

XProf is an **XLA** profiler. The framework of your job decides which
tool captures the trace and which tool shows it:

| Backend or workload | Capture with | View in |
| --- | --- | --- |
| Keras on JAX (the Kinetic default), native JAX | `jax.profiler` | XProf |
| Keras on TensorFlow | `TensorBoard(profile_batch=…)` | XProf |
| PyTorch/XLA (`torch_xla`) | `torch_xla.debug.profiler` | XProf |
| Native PyTorch (eager CUDA) | `torch.profiler` | Perfetto |

Native eager PyTorch does not compile through XLA. For that workload,
capture with `torch.profiler` and view the trace in Perfetto, not in
XProf.

## Capture a profile

Follow this pattern inside the decorated function:

:::{container} kinetic-steps
1. Run one or more warm-up steps first. The warm-up keeps the XLA
   compilation out of the trace.
2. Start the trace with `jax.profiler.trace(trace_dir)`, where
   `trace_dir` is `$KINETIC_OUTPUT_DIR/profile`.
3. Run a small number of steps inside the trace region.
4. Call `block_until_ready()` on the result before the trace region
   closes.
5. Print the trace path, so that you can find the trace later.
:::

The job needs **no extra packages**. `jax.profiler` is part of JAX, and
JAX is in the image that Kinetic builds. `xprof` is a viewer that you
install on your own machine, not a job dependency. If a profiling job
does need an extra package, add the package to a `requirements.txt` in
the directory of that script. See [Dependencies](dependencies.md).

The example below is a small JAX training loop. The same capture pattern
works in a full training job, in a KerasHub fine-tuning job, and in a
multi-host run.

```{literalinclude} ../../examples/jax_profiling_demo.py
:language: python
:caption: examples/jax_profiling_demo.py
```

The example uses `accelerator="tpu-v5litepod-1x1"`, a single TPU v5e
chip. `1x1` is the topology spelling of `tpu-v5litepod-1`; see
[Accelerators](../accelerators.md). Your cluster needs a node pool for that accelerator. If the cluster
has no such node pool, change the `accelerator=` argument to an
accelerator that the cluster has. See
[Clusters and Node Pools](clusters.md).

For **Keras on JAX**, use the same pattern. Run one warm-up epoch. Then
wrap a short `model.fit(...)` call in
`with jax.profiler.trace(trace_dir):`.

:::{note}
JAX dispatches work asynchronously. If you do not call
`block_until_ready()` inside the trace region, the trace can close before
the device work completes. The trace is then incomplete. Also keep the
window to a few steps, because a trace grows fast.
:::

For **native PyTorch**, use `torch.profiler` with
`tensorboard_trace_handler(trace_dir)`, and view the trace in Perfetto.
For **PyTorch/XLA**, use `torch_xla.debug.profiler`, which writes traces
that XProf reads.

## Where the trace goes

Kinetic sets `KINETIC_OUTPUT_DIR` in the pod to a per-job prefix in the
jobs bucket of the cluster:

```text
gs://{project}-kn-{cluster}-jobs/outputs/{job_id}
```

The example therefore writes the trace to
`gs://{project}-kn-{cluster}-jobs/outputs/{job_id}/profile`, and prints
that path in the job log. The job ID has the form `job-a1b2c3d4`. For a
detached job, `job.job_id` gives the job ID. The command `kinetic jobs
list` lists the jobs of the cluster.

Retention has two rules:

- The trace survives result collection. After a blocking call or a
  `result()` call, Kinetic deletes only the job artifacts under
  `gs://{jobs bucket}/{job_id}/`. Kinetic does not delete anything under
  `outputs/`.
- The jobs bucket deletes objects that are older than 30 days. To keep a
  trace for longer, pass `output_dir="gs://your-bucket/path"` to the
  decorator. See [Outputs and Checkpoints](checkpointing.md).

## View the trace

Install the viewer on your machine. Then point the viewer at the trace
path that the job printed:

```bash
pip install xprof gcsfs        # gcsfs lets XProf read gs:// paths directly
xprof --logdir gs://{project}-kn-{cluster}-jobs/outputs/{job_id}/profile --port 6006
# --logdir is the directory that contains plugins/ (the path that the job printed)
# then open http://localhost:6006
```

You can also copy the trace to your machine first. Run
`gcloud storage cp -r {trace-path} ./trace`. Then start XProf with
`--logdir ./trace`.

In the XProf user interface, select a tool from the tool list. Start
with **Overview Page** for the summary. Then open **Trace Viewer** for
the step timeline. Do not use the **Capture Profile** button. That
button captures a live profile from a running profiler server. In this
workflow, the job already captured the trace.

## What the tools show

- **Overview Page** — the top-level summary. It shows whether the job is
  host-bound or device-bound. Start here.
- **Trace Viewer** — the per-event timeline across the host, the TPU, and
  the GPU. Use it to find gaps and stalls.
- **Roofline** — memory-bound versus compute-bound. The result decides
  the optimization strategy.
- **Framework Op Stats** and **HLO Op Stats** — the cost by framework
  operation and by compiled HLO operation.
- **Memory Viewer** and **Memory Profile** — memory usage over time and
  at the peak. Open these tools first after an out-of-memory error.
- **Megascale Stats** — cross-slice (DCN) communication on multi-host
  [Pathways](distributed_training.md) runs.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1em` Cost Optimization
:link: cost_optimization
:link-type: doc

A profile shows where the accelerator hours go. That page shows how to
reduce the cost.
:::

:::{grid-item-card} {octicon}`database;1em` Outputs and Checkpoints
:link: checkpointing
:link-type: doc

How `KINETIC_OUTPUT_DIR` keeps your trace after the pod ends, and how to
send outputs to your own bucket.
:::

:::{grid-item-card} {octicon}`package;1em` Dependencies
:link: dependencies
:link-type: doc

How to add a package to one job with a `requirements.txt`.
:::

:::{grid-item-card} {octicon}`cpu;1em` Distributed Training
:link: distributed_training
:link-type: doc

Multi-host runs, where Megascale Stats applies.
:::
::::
