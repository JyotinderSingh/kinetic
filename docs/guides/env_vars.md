# Forward Environment Variables

Kinetic allows you to propagate local environment variables to the remote worker environment. This is useful for passing API keys, configuration, or credentials without hardcoding them in your script.

## Forwarding Variables

Use the `capture_env_vars` parameter in the `@kinetic.run()` decorator. It accepts a list of environment variable names or wildcard patterns.

```python
import kinetic


@kinetic.run(
  accelerator="tpu-v5litepod-1",
  capture_env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY", "WANDB_*"],
)
def train_model():
  import os

  # These are available in the remote process
  user = os.environ.get("KAGGLE_USERNAME")
  # ...
```

## Wildcard Support

You can use the `*` suffix to capture all environment variables that start with a specific prefix.

- `capture_env_vars=["GOOGLE_CLOUD_*"]`: Captures `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`, etc.
- `capture_env_vars=["*"]`: (Not recommended) Captures your full local environment, but not the names in the blocklist below.

### Blocklist for wildcard matches

Some variables describe your local machine, not your job. The pod applies
the captured values over its own environment. A local `PATH` or
`LD_LIBRARY_PATH` on the pod points to directories that do not exist
there, and the job then fails before your code starts. A wildcard pattern
thus never matches these names:

`PATH`, `HOME`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `LD_PRELOAD`,
`VIRTUAL_ENV`, `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, `SHELL`, `TMPDIR`,
`TEMP`, `TMP`, `HOSTNAME`, `USER`, `LOGNAME`, `SSH_AUTH_SOCK`,
`KUBERNETES_SERVICE_HOST`, `KERAS_BACKEND`

Kinetic filters only the wildcard expansion. To forward one of these
names, list the name exactly in `capture_env_vars`. For example,
`capture_env_vars=["KERAS_BACKEND"]` replaces the default backend of the
image. Kinetic writes the names that a wildcard did not capture into the
log.

## Secure Handling

Kinetic serializes the values of the requested environment variables and sends them to the remote worker as part of the job payload. Make sure that you forward only the variables that the job needs.

Kinetic writes the **names** that it captured into the log on each
submit. Kinetic never writes the values.

Kinetic also logs a warning when a captured name contains `TOKEN`,
`SECRET`, `KEY`, `PASSWORD`, or `CREDENTIAL`. The comparison ignores the
letter case. This warning is informational, and Kinetic shows it also for
a name that you listed exactly. If you intend to forward the credential,
you need no action.

The warning tells you where the value goes. Kinetic writes the value into
`payload.pkl` in the job bucket, which each job pod in the cluster can
read. Kinetic deletes the artifacts of a job when it collects a usable
result from that job. If Kinetic does not collect a result, the artifacts
stay until the lifecycle rule of the bucket deletes them. That rule
deletes objects after 30 days on a bucket that `kinetic up` created.

Use short-lived tokens for the values that you forward. For more
information, see [Security](../security.md).

## Precedence

Environment variables set via `capture_env_vars` will override any existing variables with the same name in the remote container's base environment.

## Canonical Environment Variables

Kinetic automatically sets some environment variables in the remote worker environment:

- `KINETIC_OUTPUT_DIR`: The path to the directory where outputs should be saved. By default, this is a GCS path pointing to `gs://{bucket_name}/outputs/{job_id}`. This is useful for passing to checkpointing libraries like Orbax.

> **Important**: By default, Kinetic imposes a 30-day TTL (Time to Live) on the
> GCS buckets it creates. This means anything written to the default
> `KINETIC_OUTPUT_DIR` will be automatically deleted after 30 days. If you need
> to preserve outputs longer, you should copy them to a bucket without a
> lifecycle rule or specify a custom `output_dir` pointing to a different
> location.

## Related pages

- [Configuration](../configuration.md) — full list of `KINETIC_*`
  variables and precedence rules.
- [Checkpointing](checkpointing.md) — how `KINETIC_OUTPUT_DIR` fits
  into the durable-output story.
- [LLM Fine-tuning](../examples/llm_finetuning.md) — `capture_env_vars` is the
  canonical way to forward Kaggle and other model-hub credentials.

