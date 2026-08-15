# Outputs and Checkpoints

The filesystem of the pod is temporary. When your function returns, or
when the pod stops for another reason, every file on the pod is lost.
Kinetic gives each job an **output directory**: a Cloud Storage prefix
that stays after the pod stops. The pod sees that prefix as the
environment variable `KINETIC_OUTPUT_DIR`. This page explains what to
write where and how to set the output directory. It also explains how to
resume a training run from a checkpoint and how long the files stay.

## A first job that writes outputs

Kinetic sets `KINETIC_OUTPUT_DIR` in the pod. Read the variable and write
under that path. If the variable is not present, use a local path. Then
the same function also works when you call it locally for a test:

```python
import os

import kinetic


@kinetic.run(accelerator="cpu")
def train():
  # Remote: KINETIC_OUTPUT_DIR is gs://.../outputs/<job_id>.
  # Local: fall back to a path under /tmp for a direct test call.
  output_dir = os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/local_checkpoints")
  # ... train, and write checkpoints and artifacts under output_dir ...
  return f"saved to {output_dir}"
```

Every call of `train()` is a new job with a new job ID. By default, the
output directory contains the job ID, so every call also gets a new,
empty output directory. That default is correct for a job that runs one
time. If a second call must find the files of the first call, both calls
must use the same output directory. A resume from a checkpoint is one
example. Pass `output_dir=` to the decorator:

```python
@kinetic.run(accelerator="cpu", output_dir="gs://my-bucket/runs/exp-01")
def train(): ...
```

See [Resume a job from a checkpoint](#resume-a-job-from-a-checkpoint)
below.

## Three kinds of artifact

A Kinetic job produces three kinds of artifact. Each kind has its own
location and its own lifecycle:

| Artifact | What it is | Where it goes |
| -------- | ---------- | ------------- |
| Job return value | The Python value that your function returns | `gs://{jobs bucket}/{job_id}/result.pkl`, then your local process |
| Durable outputs | Files that you write during the run | `KINETIC_OUTPUT_DIR` in Cloud Storage |
| Resumable checkpoints | Periodic snapshots of the training state | A fixed subdirectory under `KINETIC_OUTPUT_DIR` |

Use the return value for **small** results: a final loss, a dict of
metrics, a path string. Write large files under the output directory.
Write checkpoints to a fixed subdirectory under the output directory, so
that a later run finds them at a known path.

## The default output directory

Kinetic sets `KINETIC_OUTPUT_DIR` when the job starts. By default, the
variable points to a per-job prefix in the jobs bucket of your cluster:

```text
gs://{project}-kn-{cluster}-jobs/outputs/{job_id}
```

`{project}` and `{cluster}` come from the active profile, unless you
override them for the job. `{job_id}` is the ID of the job, for example
`job-3f9a1c2b`. `kinetic up` creates the jobs bucket one time, and every
job on that cluster uses the same bucket.

The pod runs as the node service account of the cluster,
`kn-{cluster}-nodes@{project}.iam.gserviceaccount.com`. That account can
read and write the jobs bucket. It has no access to other buckets unless
you grant that access.

## Set the output directory

Pass `output_dir="gs://..."` to `@kinetic.run()` to replace the default
location for a job.

For a script that you cannot edit, export `KINETIC_OUTPUT_DIR` in your
local shell before you submit the job. Kinetic reads the local variable
at submit time and passes the value to the pod. The decorator argument
wins over the local environment variable, and the local environment
variable wins over the default. See [Configuration](../configuration.md)
for the full precedence rules. `kinetic config` shows the value of
`KINETIC_OUTPUT_DIR` if you set the variable in your shell.

You cannot change the output directory from the `kinetic jobs` commands.
The output directory is a property of the job that you set at submit
time.

:::{note}
If `output_dir=` points to a bucket that `kinetic up` did not create, the
pod cannot write there until you grant access. Give the node service
account of the cluster the `roles/storage.objectAdmin` role and the
`roles/storage.legacyBucketReader` role on that bucket:

```bash
gcloud storage buckets add-iam-policy-binding gs://my-bucket \
  --member=serviceAccount:kn-kinetic-cluster-nodes@my-project.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding gs://my-bucket \
  --member=serviceAccount:kn-kinetic-cluster-nodes@my-project.iam.gserviceaccount.com \
  --role=roles/storage.legacyBucketReader
```

Replace `kinetic-cluster` and `my-project` with the cluster name and the
project of your profile. Orbax and TensorStore need the second role to
read the bucket metadata.
:::

## Resume a job from a checkpoint

A job can stop before the work is done: Google Cloud preempts a Spot
node, a node fails, or your code raises an error. Kinetic does not submit
the job again for you. The checkpoints that the job wrote stay in Cloud
Storage. To continue the work, submit the function again with the
**same** output directory. Your code then finds the latest checkpoint
under that directory and continues from that checkpoint.

The output directory must be the same for each call. With the default
output directory, each call gets a new job ID and therefore a new, empty
prefix. The second call then starts from step 0. There are two ways to
set a fixed directory:

1. Pass `output_dir=` to the decorator:

   ```python
   @kinetic.run(
     accelerator="tpu-v5litepod-4",
     output_dir="gs://my-bucket/runs/exp-01",
   )
   def train(): ...


   train()  # writes checkpoints under gs://my-bucket/runs/exp-01
   train()  # finds them and resumes
   ```

2. Export `KINETIC_OUTPUT_DIR` before both submissions:

   ```bash
   export KINETIC_OUTPUT_DIR=gs://my-bucket/runs/exp-01
   python train.py   # first run
   python train.py   # resumes
   ```

A path in the jobs bucket also works, for example
`gs://my-project-kn-kinetic-cluster-jobs/outputs/exp-01`, and needs no
extra access grant. The 30-day rule of that bucket applies (see below).

## Recommended directory layout

The layout below works for a single job and for many jobs:

```text
$KINETIC_OUTPUT_DIR/
├── checkpoints/        # Orbax / model.save_weights — periodic snapshots
├── logs/               # extra logs that your code writes (stdout already streams)
├── metrics/            # TensorBoard / JSON metric dumps
└── final/              # post-training artifacts: exported model, eval results
```

Use the subdirectories that fit your workflow. Kinetic does not read or
interpret the layout. Kinetic only requires that you write under the
prefix that it gives you.

## Retention and cleanup

**The 30-day rule.** The jobs bucket has a lifecycle rule that deletes
every object 30 days after its creation. Cloud Storage therefore deletes
the files under the default `KINETIC_OUTPUT_DIR` after 30 days. That
default fits short experiments. If a checkpoint or a model must stay
longer than 30 days, do one of these two things:

- Copy the files to a bucket without a lifecycle rule, with
  `gcloud storage cp` or the Cloud Storage client library.
- Set `output_dir=` to a bucket that you manage, with the lifecycle
  rules that you choose (see the access note above).

**Job cleanup does not delete outputs.** A blocking call, a
`JobHandle.result()` call, `JobHandle.cleanup(gcs=True)`, and
`kinetic jobs cleanup JOB_ID` delete the job artifacts under
`gs://{jobs bucket}/{job_id}/`. Those artifacts include the serialized
function, the source archive, and the result. These calls never delete
files under `KINETIC_OUTPUT_DIR`. `result()` deletes the artifacts only
after it collects a usable result; a failed job keeps its artifacts. Pass
`cleanup=False` to `result()`, or `--no-cleanup` to
`kinetic jobs result`, to keep the artifacts of a job that succeeded.

:::{warning}
`kinetic down` deletes the cluster and the jobs bucket, with every output
in that bucket. Before you run `kinetic down`, copy the files that you
want to keep to a bucket that Kinetic does not manage.
:::

## Checklist for a long job

Follow these steps for a job that you do not want to run again from the
start:

:::{container} kinetic-steps
1. **Read `KINETIC_OUTPUT_DIR`** inside the function, and write every
   durable file under that path.

2. **Write checkpoints to a fixed subdirectory**, for example
   `$KINETIC_OUTPUT_DIR/checkpoints/`, so that you know the resume path.

3. **Choose a checkpoint interval** that limits the work that a restart
   loses: every N steps, or every M minutes.

4. **Set a fixed output directory** with `output_dir=` or with a local
   `KINETIC_OUTPUT_DIR`. Without a fixed directory, the second submission
   gets a new, empty prefix and cannot resume.

5. **Test the resume before the long run.** Submit the same function two
   times with the same output directory, and confirm that the second
   call continues from the last checkpoint.

6. **Copy the final artifacts** to a bucket without the 30-day rule if the
   run is important.
:::

## JAX example

The example below shows the write-and-read pattern with Orbax. The
function points a `CheckpointManager` at `KINETIC_OUTPUT_DIR`. It
restores the latest step if one exists, and it saves a checkpoint after
each step.

The example uses `@kinetic.run(accelerator="cpu")` with no `output_dir=`,
so each call gets a new default output directory and starts from step 0.
To see a real resume, add `output_dir="gs://..."` to the decorator, or
export `KINETIC_OUTPUT_DIR` before you run the script two times.

```{literalinclude} ../../examples/example_checkpoint.py
```

## Keras example

The Keras example uses the same pattern. `model.get_weights()` returns a
list of NumPy arrays, and Orbax saves that list as a PyTree. After a
restore, `model.set_weights()` loads the arrays back into the model. As
with the JAX example, add `output_dir=` or export `KINETIC_OUTPUT_DIR`
to make two calls share one output directory.

```{literalinclude} ../../examples/example_keras_checkpoint.py
```

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1em` Working with Data
:link: data
:link-type: doc

The input side: ship local files and read Cloud Storage data from your
function.
:::

:::{grid-item-card} {octicon}`clock;1em` Detached Jobs
:link: async_jobs
:link-type: doc

Submit long jobs with `run_async()`, and collect or clean up their
results later.
:::

:::{grid-item-card} {octicon}`graph;1em` Cost Optimization
:link: cost_optimization
:link-type: doc

Spot capacity lowers the cost, and a checkpoint makes a preempted run
resumable.
:::
::::
