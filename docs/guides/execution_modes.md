# Execution Modes

Kinetic ships three ways of producing the container that runs your job. The
mode you pick controls how long the first run takes, how much you can change
between runs without paying a build cost, and how much of the image you own.

The three modes:

- **Bundled mode** — Kinetic builds a custom image with your dependencies
  baked in, via Cloud Build. This is the default.
- **Prebuilt mode** — Kinetic pulls a published base image and installs your
  dependencies at pod startup with `uv pip install`.
- **Custom image mode** — You provide a full image URI; Kinetic skips both
  the build and the install steps.

You select the mode with the `container_image` argument on `@kinetic.run()`:

```python
@kinetic.run(accelerator="tpu-v6e-8")                              # bundled (default)
@kinetic.run(accelerator="tpu-v6e-8", container_image="bundled")   # bundled (explicit)
@kinetic.run(accelerator="tpu-v6e-8", container_image="prebuilt")  # prebuilt
@kinetic.run(accelerator="tpu-v6e-8", container_image="us-docker.pkg.dev/me/repo/img:v1")  # custom
```

:::{tip}
**Recommended default:** **bundled mode**. It's the only mode that works
out of the box, and it's the right choice for any workflow where your
dependencies are reasonably stable. Cached images make warm runs fast;
the build step only re-runs when your deps change.

Reach for **prebuilt mode** only if you're iterating on `requirements.txt`
several times a day and the per-iteration build cost is hurting you — and
note that prebuilt currently requires you to publish your own base image
with `kinetic build-image`, since no blessed base images ship with Kinetic
today.
:::

## Recommendation matrix

You are…                                           | Use                             | Why
-------------------------------------------------- | ------------------------------- | ----------------------------------------------------------------
A first-time user                                  | **bundled**                     | The only mode that works without publishing your own base image.
Iterating quickly on the same code                 | **bundled**                     | The dep-hashed image is cached; warm runs start in seconds.
Changing dependencies multiple times a day         | **prebuilt**\*                  | Skip the rebuild — install runs at pod startup instead.
Running with a large dependency set                | **bundled**                     | Pay the install cost once at build time, not on every run.
Producing a reproducible production run            | **bundled**                     | The exact environment is frozen into a tagged image.
Needing custom system libs (CUDA builds, C++ deps) | **custom image**                | Bundled and prebuilt can't add system packages.
Pulling private packages                           | **bundled** or **custom image** | Bundled rebuilds on dep changes; custom gives full control.
On a corporate base image                          | **custom image**                | Use whatever your platform team blesses.

\* Prebuilt mode requires a base image at the configured repo. Kinetic does
not currently ship blessed base images, so you'll need to run
`kinetic build-image` once and set `KINETIC_BASE_IMAGE_REPO` before this is a
practical option.

## Bundled mode

Bundled mode runs Cloud Build to produce a tagged image with your project's
dependencies installed. The image tag is a hash of those dependencies, so two
jobs with the same `requirements.txt` reuse the same cached image.

```python
@kinetic.run(accelerator="tpu-v6e-8")
def train():
  import keras

  ...
```

**Startup expectations:**

- **Cold (first run, or after a dep change):** ~2–5 minutes for the build.
- **Warm (cached image):** under a minute to schedule and start the pod.

**Use it when:** any time you don't have a strong reason to do something else.
This is the recommended default.

**Avoid it when:** you change `requirements.txt` several times a day and the
2–5 minute rebuilds are dominating your cycle time — at that point prebuilt
mode is worth the setup cost.

## Prebuilt mode

Prebuilt mode pulls a published base image
(`{repo}/base-{cpu|gpu|tpu}:{kinetic-version}`) that already contains the
accelerator runtime, then runs `uv pip install` against your project's
dependencies at pod startup.

```python
@kinetic.run(accelerator="tpu-v6e-8", container_image="prebuilt")
def train(): ...
```

:::{warning}
**You need to publish a base image first.** Kinetic does not currently ship
blessed prebuilt base images. Before you can use prebuilt mode, run
`kinetic build-image --repo <your-repo>` once and set
`KINETIC_BASE_IMAGE_REPO=<your-repo>` (or pass `base_image_repo=` to the
decorator). Kinetic builds the image tag from the version of your
installed Kinetic client. If you upgrade the client, publish a base image
with the new tag first. See [Container Images](containers.md) for the full
workflow.
:::

**Startup expectations:**

- **Image pull:** typically 30–60 seconds the first time on a node, near zero
  once cached.
- **Dependency install:** scales with the size of your `requirements.txt` —
  small projects start in under a minute, large ones a few minutes.

**Use it when:** you've published a base image and you're churning
`requirements.txt` often enough that bundled rebuilds are slowing you down.

**Avoid it when:** every job has a long install step — bundled amortizes that
cost into a single build, and once a bundled image is cached, warm runs are
faster than prebuilt.

## Custom image mode

Pass a full image URI and Kinetic uses it as-is. No build, no install — your
image is responsible for every dependency your function needs.

```python
@kinetic.run(
  accelerator="tpu-v6e-8",
  container_image="us-docker.pkg.dev/my-project/kinetic/my-image:v1.0",
)
def train(): ...
```

**Requirements:** the image must

- Include the Kinetic runner script at `/app/remote_runner.py`. Kinetic
  invokes the container with `python3 -u /app/remote_runner.py`, which
  overrides whatever `ENTRYPOINT` or `CMD` the image declares — the only
  hard requirement is that the file is present at that path.
- Have `python3` on `PATH`, with a version compatible with the one you
  used to pickle the function locally.
- Install `cloudpickle`, `google-cloud-storage`, and `absl-py` — the
  runner imports them directly.
- Install whatever other libraries your function imports.
- It's also recommended to install the `kinetic` package itself if your
  function (or anything it imports) references it. The runner doesn't
  need it, but user code often does.
- Be pullable from your GKE nodes (Artifact Registry in the same GCP
  project, or a public registry).

**Startup expectations:** a single image pull, then immediate execution.
Cold pulls vary widely with image size and registry latency.

**Use it when:** you have system libraries that bundled or prebuilt can't add,
you need a corporate-vetted base image, or you want full control over the
image lifecycle.

## How Kinetic decides what to build or install

The dispatch happens at job submit time inside the backend execution path
(`kinetic/backend/execution.py`):

1. If `container_image == "prebuilt"`, Kinetic resolves the prebuilt base
   image for your accelerator category and uploads your filtered
   `requirements.txt` to GCS for runtime install.
2. Else if `container_image` is `None` or `"bundled"`, Kinetic packages your
   working directory, computes a dependency hash, and either reuses a cached
   image or runs Cloud Build to produce a new one.
3. Otherwise, Kinetic treats `container_image` as a literal image URI and
   uses it directly — no packaging of dependencies, no install.

In all three modes, two more things happen regardless of which mode you
picked:

- **Your function and its captured closures.** Kinetic pickles them with
  `cloudpickle` and uploads them to Cloud Storage. Kinetic serializes the
  modules of your own project by value, so the pod does not import them.
  The runner in the pod downloads the payload, unpickles your function,
  and calls it.
- **Your project source.** Kinetic zips it into a `context.zip` file and
  uploads that file to Cloud Storage. The archive starts at the *package
  root*, not at the directory of the script that you ran. Kinetic first
  walks up out of every directory that holds an `__init__.py` file.
  Kinetic then walks up to the nearest directory that holds a
  `pyproject.toml`, `requirements.txt`, `setup.py`, `setup.cfg`, or `.git`
  entry. Kinetic excludes the paths in your `Data(...)` objects and the
  default exclusion list (`.venv`, `node_modules`, and the cache
  directories).

The runner extracts the archive into the workspace of the pod. The runner
then rebuilds `sys.path` and changes to the workspace directory that
matches your client working directory. Your imports and your relative-path
reads thus operate as they do on your machine. This is most important in
**custom image mode**: the image supplies the installed packages, but
Kinetic still ships your project source with the job. You do not put the
source into the image.

[What Ships to the Pod](packaging.md) gives the full contract:

- Root detection.
- Exclusions and `.kineticignore`.
- The rules for the working directory and for `sys.path`.
- The guarantees for argument types.
- Version matching between your client and the pod.

## Related pages

- [What Ships to the Pod](packaging.md) — the full packaging contract.
- [Dependencies](dependencies.md) — how Kinetic discovers what to install.
- [Container Images](containers.md) — base-image workflow and
  `kinetic build-image`.
- [Getting Started](../getting_started.md) — your first run end-to-end.
