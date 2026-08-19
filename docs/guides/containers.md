# Container Images

Every job runs inside a container image. By default, Kinetic builds that
image for you and caches it. Most people never change this default. This
page is for three cases where the default does not fit. The build step
is too slow for your workflow. You need system libraries that a Python
install cannot provide. Or your organization requires a vetted base
image.

## The three modes

The `container_image` argument of `@kinetic.run()` selects how Kinetic
produces the image:

| Mode | `container_image=` | Build step | Where your dependencies install |
| ---- | ------------------ | ---------- | ------------------------------- |
| **Bundled** (default) | `None` or `"bundled"` | Cloud Build, cached by a hash of the inputs | In the image, at build time |
| **Prebuilt** | `"prebuilt"` | None. Kinetic pulls a base image that you published. | In the pod, at start, with `uv pip install` |
| **Custom** | An image URI | None. You build and push the image. | In your image. Kinetic installs nothing. |

```python
@kinetic.run(accelerator="tpu-v5litepod-4")  # bundled (the default)
def train_bundled(): ...


@kinetic.run(accelerator="tpu-v5litepod-4", container_image="prebuilt")
def train_prebuilt(): ...


@kinetic.run(
  accelerator="tpu-v5litepod-4",
  container_image="us-docker.pkg.dev/me/repo/img:v1",  # custom
)
def train_custom(): ...
```

In all three modes, Kinetic ships your function and your project source
with the job. The image supplies the installed packages only. See
[What Ships to the Pod](packaging.md).

## Which mode to use

| Situation | Mode | Reason |
| --------- | ---- | ------ |
| You start with Kinetic | Bundled | Works without setup. |
| Your dependencies change less than once a day | Bundled | The cached image removes the build from later runs. |
| Your dependency set is large | Bundled | You pay the install time one time, at build time. |
| You need a reproducible environment | Bundled | The exact environment is frozen in a tagged image. |
| You change the dependency file many times a day | Prebuilt | No build. The install runs at pod start. Needs a published base image. |
| You need system libraries (custom CUDA builds, C++ libraries) | Custom, or prebuilt with your own Dockerfile | Bundled installs Python packages only. `kinetic build-image --dockerfile` lets a prebuilt base image contain system libraries. |
| You need private packages that require credentials at install time | Custom | You control the build environment. |
| Your organization requires a vetted base image | Custom | Use the image that your platform team approves. |

## Bundled mode (default)

Kinetic runs Cloud Build to produce an image and pushes the image to the
Artifact Registry repository of the cluster. The image contains:

- A `python:{X.Y}-slim` base image, where `X.Y` is the Python minor
  version of your local interpreter. Pickled code is not portable across
  minor versions, so this rule keeps the pod compatible with your client.
- JAX with the runtime for the accelerator category: `jax[tpu]` with
  `libtpu`, `jax[cuda12]`, or plain `jax` for CPU.
- `keras`, `cloudpickle`, `google-cloud-storage`, and `keras-kinetic`
  pinned to your client version.
- The packages from your dependency file, without the JAX entries. See
  [Dependencies](dependencies.md).
- The Kinetic runner script.

Kinetic tags the image with a hash of the base image, the accelerator
category, the Kinetic version, the filtered dependency file, the runner
script, and the Dockerfile template. Two jobs with the same inputs share
one image.

**Timing:**

- **Cold** (first run, or after a change to the dependency file): about
  5 to 10 minutes for the build.
- **Warm** (cached image): no build. The pod starts in less than 1
  minute while a node still runs, or after the 2 to 5 minutes that a new
  node needs.

To inspect a build, list the Cloud Build history of the project:

```bash
gcloud builds list --limit=5
gcloud builds log <build-id>
```

## Prebuilt mode

In prebuilt mode, Kinetic does not build. Kinetic pulls a **base image**
that already contains the accelerator runtime and the core packages, and
the pod installs your dependency file at start with `uv pip install`.

:::{warning}
**Kinetic does not publish base images.** The default repository name is
`kinetic` on Docker Hub, but no images exist there. Before you use
prebuilt mode, publish base images to your own repository with
`kinetic build-image`, and point Kinetic at that repository.
:::

### Set up prebuilt mode

1. Build and push the base images. One image per accelerator category:

   ```bash
   kinetic build-image --repo us-docker.pkg.dev/my-project/kinetic-base
   ```

2. Tell Kinetic where the images are. Set the environment variable, or
   pass the repository in the decorator:

   ::::{tab-set}

   :::{tab-item} Environment variable

   ```bash
   export KINETIC_BASE_IMAGE_REPO=us-docker.pkg.dev/my-project/kinetic-base
   ```
   :::

   :::{tab-item} Decorator argument

   ```python
   @kinetic.run(
     accelerator="gpu-l4",
     container_image="prebuilt",
     base_image_repo="us-docker.pkg.dev/my-project/kinetic-base",
   )
   def train(): ...
   ```
   :::

   ::::

3. Select the mode with `container_image="prebuilt"`.

### How a prebuilt job starts

1. Kinetic resolves the image name
   `{repo}/base-{cpu|gpu|tpu}:{kinetic version}`. The version is the
   version of your installed `keras-kinetic` package. If you upgrade the
   client, run `kinetic build-image` again to publish images with the new
   tag.
2. Kinetic filters the JAX entries out of your dependency file and
   uploads the result next to the job artifacts. Kinetic refuses a line
   that points to a local path (`-r other.txt`, `-e .`, `./wheel.whl`),
   because that path does not exist on the pod.
3. The pod pulls the base image, runs `uv pip install` on the uploaded
   file, and then runs your function.

**Timing:** the image pull takes 30 to 60 seconds the first time on a
node, and almost no time once the node has the image. The install time
depends on your dependency file. A small file installs in less than 1
minute.

The base image must have the same Python minor version as the client
that submits the job. `kinetic build-image` uses the Python version of
the machine that runs it.

## Custom image mode

Pass an image URI, and Kinetic uses the image without changes. Kinetic does not
build and does not install. Your image is responsible for every package
that your function imports.

```python
@kinetic.run(
  accelerator="tpu-v5litepod-4",
  container_image="us-docker.pkg.dev/my-project/kinetic/my-image:v1.0",
)
def train(): ...

```

The image must satisfy these requirements:

* The Kinetic runner script is at `/app/remote_runner.py`. Kinetic
starts the container with `python3 -u /app/remote_runner.py`, which
replaces the `ENTRYPOINT` and `CMD` of the image. Copy the script from
the `kinetic/runner/` directory of the installed package.
* `python3` is on `PATH`, with the same minor version as your client.
* The packages `cloudpickle`, `google-cloud-storage`, and `absl-py` are
installed. The runner imports them.
* Every other package that your function imports is installed.
* We recommend that `keras-kinetic` is installed. The runner does not
need it, but user code often imports it.
* The GKE nodes can pull the image: Artifact Registry in the same
project, or a public registry.

Kinetic still ships your project source with the job and extracts it on
the pod. Do not copy your source into the image.

**Timing:** one image pull, then the function runs. The pull time depends
on the image size and the registry.
## `kinetic build-image`

`kinetic build-image` builds base images with Cloud Build and pushes them
to Docker Hub or to Artifact Registry. The command builds one image per
accelerator category. The command needs an existing cluster: it uploads
the build context to the builds bucket of the cluster and runs Cloud
Build as the build service account of the cluster. Run `kinetic up`
first.

```bash
# Interactive: the command asks for the registry and the settings.
kinetic build-image

# Artifact Registry, without prompts.
kinetic build-image \
  --repo us-docker.pkg.dev/my-project/kinetic-base \
  --project my-project \
  --yes

# GPU and TPU images only.
kinetic build-image --repo myuser/kinetic --category gpu --category tpu

# A custom Dockerfile.
kinetic build-image --repo myuser/kinetic --dockerfile ./Dockerfile.custom

# A specific tag. The default is the kinetic package version.
kinetic build-image --repo myuser/kinetic --tag v2.0.0
```

| Option                 | Description                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `--repo`               | Image repository (Docker Hub or Artifact Registry). Omit to select interactively.                                              |
| `--category`           | Accelerator categories to build: `cpu`, `gpu`, `tpu` (default: all). Repeatable.                                               |
| `--tag`                | Image version tag (default: kinetic package version).                                                                          |
| `--dockerfile`         | Path to a custom Dockerfile. The Dockerfile must install `uv`, `cloudpickle`, `google-cloud-storage`, and `absl-py`. It must also copy `remote_runner.py` to `/app/`. |
| `--update-credentials` | Re-enter Docker Hub credentials even if they already exist in Secret Manager.                                                  |
| `--yes`, `-y`          | Skip confirmation prompt.                                                                                                      |
| `--project`            | GCP project ID (default: `KINETIC_PROJECT`).                                                                                   |
| `--cluster`            | GKE cluster name (default: `kinetic-cluster`).     

Registry notes:

- **Docker Hub** — the command asks for your Docker Hub username and an
  access token on first use, stores them in Secret Manager, and Cloud
  Build uses them for the push.
- **Artifact Registry** — no extra credentials. The build service
  account pushes directly. The command prints the `gcloud` commands that
  create the repository and grant the permissions.

## How Kinetic decides

At submit time, Kinetic reads `container_image`:

1. `"prebuilt"` — resolve the base image for the accelerator category,
   filter and upload the dependency file.
2. `None` or `"bundled"` — hash the inputs, then reuse the cached image
   or run Cloud Build.
3. Any other string — use it as an image URI. No build, no install.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1em` Dependencies
:link: dependencies
:link-type: doc

How Kinetic finds the dependency file, and what it filters out.
:::

:::{grid-item-card} {octicon}`file-directory;1em` What Ships to the Pod
:link: packaging
:link-type: doc

The function, the source archive, and the Python version rule.
:::

:::{grid-item-card} {octicon}`graph;1em` Cost Optimization
:link: cost_optimization
:link-type: doc

Cloud Build charges and how the image cache limits them.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

Build failures and version-skew errors.
:::
::::
