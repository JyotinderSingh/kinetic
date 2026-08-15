# Forward Environment Variables

The pod does not see the environment variables of your shell. When your
function needs a value from your local environment — an API key, a
Kaggle credential, a configuration flag — list the variable name in
`capture_env_vars`. Kinetic copies the value into the pod at submit
time. This page covers that parameter, its wildcard rules, and how to
handle secrets.

This page is not about the `KINETIC_*` variables that select the
project, the zone, the cluster, or the namespace. See
[Configuration](../configuration.md) for those.

## Forward a variable

Pass a list of names or patterns to `capture_env_vars`:

```python
import kinetic


@kinetic.run(
  accelerator="tpu-v5litepod-4",
  capture_env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY", "WANDB_*"],
)
def train_model():
  import os

  user = os.environ.get("KAGGLE_USERNAME")  # the value from your shell
  ...
```

Kinetic reads the values when you call the function, stores them in the
job payload, and sets them in the pod before your function runs. A
captured value replaces a variable with the same name in the image.

## Wildcards

A name that ends with `*` is a prefix pattern:

- `capture_env_vars=["GOOGLE_CLOUD_*"]` captures `GOOGLE_CLOUD_PROJECT`,
  `GOOGLE_CLOUD_REGION`, and every other name with that prefix.
- `capture_env_vars=["*"]` captures your full local environment, except
  the blocklist below. We do not recommend this pattern, because it also
  captures every secret in your shell.

A `*` in any other position is not a wildcard.

### Names that a wildcard never matches

Some variables describe your machine, not your job. A local `PATH` or
`LD_LIBRARY_PATH` points to directories that do not exist in the pod,
and the job fails before your code starts. A wildcard pattern therefore
never matches these names:

`PATH`, `HOME`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `LD_PRELOAD`,
`VIRTUAL_ENV`, `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, `SHELL`, `TMPDIR`,
`TEMP`, `TMP`, `HOSTNAME`, `USER`, `LOGNAME`, `SSH_AUTH_SOCK`,
`KUBERNETES_SERVICE_HOST`, `KERAS_BACKEND`

The filter applies to wildcard matches only. To forward one of these
names, list the name exactly. For example,
`capture_env_vars=["KERAS_BACKEND"]` replaces the default Keras backend
of the image. Kinetic logs the names that a wildcard skipped.

## Secrets

Kinetic stores the captured values in `payload.pkl` in the jobs bucket.
Every job pod in the cluster can read that bucket. Kinetic deletes the
payload when it collects a usable result with the default cleanup. In
three cases the payload stays until you call `cleanup()` or until the
30-day lifecycle rule of the bucket deletes it: Kinetic collects no
result, you pass `cleanup=False`, or you use `debug=True`. Because of this:

- Forward only the variables that the job needs.
- Use short-lived tokens where you can.

Kinetic logs the **names** that it captured on each submit. Kinetic never
logs the values. Kinetic also logs a warning when a captured name contains
`TOKEN`, `SECRET`, `KEY`, `PASSWORD`, or `CREDENTIAL`, in any letter case.
The warning is informational and appears also for a name that you listed
exactly. If you intend to forward the credential, no action is needed.
See [Security](../security.md).

## Variables that Kinetic sets in the pod

Kinetic sets `KINETIC_OUTPUT_DIR` in every pod. The value is a Cloud
Storage location, by default `gs://{jobs bucket}/outputs/{job_id}`. Write
every file that you want to keep under that location. See
[Outputs and Checkpoints](checkpointing.md).

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`gear;1em` Configuration
:link: ../configuration
:link-type: doc

The `KINETIC_*` variables that configure Kinetic itself.
:::

:::{grid-item-card} {octicon}`shield;1em` Security
:link: ../security
:link-type: doc

The trust model, and where the payload lives.
:::

:::{grid-item-card} {octicon}`key;1em` LLM Fine-tuning
:link: ../examples/llm_finetuning
:link-type: doc

`capture_env_vars` for Kaggle and other model-hub credentials.
:::
::::
