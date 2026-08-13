# Managing Dependencies

There are three independent things going on when Kinetic runs your job:

1. **Dependency discovery** — Kinetic figures out which packages your
   project needs by reading `requirements.txt` or `pyproject.toml` from
   your [entry directory](packaging.md#the-package-root) or from a
   directory above it.
2. **Container mode choice** — those dependencies either get baked into
   a custom image (bundled mode), installed at pod startup (prebuilt
   mode), or ignored entirely (custom image mode). See
   [Execution Modes](execution_modes.md).
3. **JAX filtering** — accelerator runtime packages (`jax`, `jaxlib`,
   `libtpu`) are filtered out before install so they don't shadow the
   hardware-correct versions in the container.

This page focuses on (1) and (3). (2) lives on its own page:
[Execution Modes](execution_modes.md).

## A first run

Drop a `requirements.txt` next to your script and Kinetic picks it up
automatically:

```text
# requirements.txt
keras
numpy
pandas
```

```python
@kinetic.run(accelerator="tpu-v6e-8")
def train():
  import pandas as pd  # installed automatically on the remote

  ...
```

`pyproject.toml` works equally well — Kinetic reads
`[project.dependencies]`. If both files exist, `requirements.txt` wins.

:::{tip}
**Recommended defaults:**

- Pin only the libraries you actually depend on. The fewer packages, the
  faster your image builds (or your prebuilt-mode pod start).
- Don't pin `jax`, `jaxlib`, `libtpu`, or any other accelerator runtime
  — Kinetic filters them out and uses the version in the container.
- Use a `pyproject.toml` if you already have one for local development
  rather than maintaining a separate `requirements.txt`.
:::

## How discovery works

When you call a decorated function, Kinetic starts at the entry
directory. Kinetic then walks **up** one directory at a time. At each
directory, Kinetic does these steps:

1. If a file with the name `requirements.txt` is there, use that file.
2. If not, and a file with the name `pyproject.toml` is there, read
   `[project.dependencies]` from that file.
3. If not, walk up one directory and do the steps again.

Kinetic examines files only. If a directory has the name
`requirements.txt` or `pyproject.toml`, Kinetic ignores that directory
and continues the walk.

The walk has bounds, so that it cannot leave your project and take a
foreign file:

- Kinetic stops after it examines the first directory that holds a `.git`
  entry. The entry can be a directory or a file, because a git worktree
  uses a file. Kinetic examines that directory for a dependency file
  before the walk stops.
- Kinetic stops at your home directory and at the root of the file
  system. Kinetic examines these directories too.

If Kinetic finds no dependency file, the job gets only the packages of the
base image.

Kinetic writes the name of the file that it selected into the log
(`Using dependency file: ...`) on each submit. If the installed packages
are not the packages that you expected, read this log line first. If both
files are in the selected directory, the log also names the file that
Kinetic selected. Kinetic also logs a warning when it finds no `.git`
marker and the selected file is not in your entry directory. Kinetic thus
never takes a foreign file without a message.

:::{warning}
A `pyproject.toml` file with no `[project.dependencies]` table still stops
the walk. If your dependencies are in a `requirements.txt` file above it,
Kinetic does not find them. Kinetic logs a warning that the selected file
declares no dependencies, and that the job will get only the base-image
packages. Move or copy your dependency file into your entry directory.

Kinetic reads `[project.dependencies]` only. Kinetic logs a warning when
it finds dependencies in `[tool.poetry.dependencies]`, in
`[project.optional-dependencies]`, in `[dependency-groups]`, or under
`dynamic = ["dependencies"]`.
:::

Kinetic ships only the [package root](packaging.md) in `context.zip`.
Kinetic can thus select a dependency file above the package root. Kinetic
installs the packages from that file, but the directory of that file is
not on the pod.

The image build and the pod see the generated requirements file alone.
A line that points to a path on your machine (`-r base.txt`, `-e .`, or
`./local-wheel`) thus cannot resolve. In prebuilt mode, Kinetic refuses
such a line at submit time and names the line that it refuses. In bundled
mode, Kinetic does not make this check, and the install fails inside Cloud
Build instead.

In bundled mode, the discovered file is hashed and used as part of the
image cache key — change the file, and the next run rebuilds. In
prebuilt mode, the same file is uploaded and installed at pod startup.
In custom image mode, the file is ignored entirely.

## JAX and accelerator runtimes

Kinetic's bundled and prebuilt images already have `jax`, `jaxlib`, and
the right accelerator backend (`libtpu` on TPU, CUDA libs on GPU)
installed and pinned to versions that match the container. To prevent
your `requirements.txt` from clobbering that, Kinetic strips these
entries before install:

- `jax`
- `jaxlib`
- `libtpu`
- `libtpu-nightly`

If you have a specific reason to override the in-container JAX —
testing a new release, reproducing a bug — append `# kn:keep` to the
line:

```text
jax==0.4.25 # kn:keep
jaxlib==0.4.25 # kn:keep
```

This works in `requirements.txt`. Use it sparingly; getting JAX +
`jaxlib` + accelerator runtime versions to line up by hand is a known
source of obscure crashes.

## Private packages

Bundled-mode builds install your dependencies inside Cloud Build. Cloud
Build does not inherit your local `pip.conf`, environment variables, or
shell credentials, so anything the installer needs in order to find or
authenticate to a private index has to be present in the project source
that gets uploaded to the build.

You have two practical options:

::::{tab-set}

:::{tab-item} Bundled mode with an index URL

Add `--index-url` or `--extra-index-url` as a line in
`requirements.txt`. The installer reads these directives and uses them
when resolving every package in the file:

```text
--extra-index-url https://my-org-private-index.example.com/simple
my-private-package==1.2.3
some-public-dep==2.0.0
```

This works without extra setup if the index is publicly reachable
(no auth required), or if it sits behind network ACLs that the Cloud
Build pool already satisfies (for example, a GCP-internal Artifact
Registry repo that the build service account has read access to).
:::

:::{tab-item} Custom image mode

If your private packages need credentials at install time, system
libraries, or unusual build flags, prebuild a container image with
them installed and pass it as `container_image="<your-image-uri>"`.
This gives you full control over the build environment, including
`pip.conf`, secret mounts, and `gcloud` authentication. See
[Container Images](containers.md).
:::

::::

:::{warning}
Avoid embedding secrets in `requirements.txt`
(`https://user:token@host/...`); the file is uploaded to GCS and used
as part of the build context, so any credentials it contains will end
up in build logs and cached artifacts.
:::

## Common dependency pitfalls

- **Pinning `jax` without `# kn:keep`** — the pin is silently dropped
  and you get the in-container version anyway. If you actually want a
  pin, use `# kn:keep`. If you don't, drop the line.
- **Listing TensorFlow alongside JAX** — both ship their own copy of
  the accelerator runtime. They can co-exist, but on TPU you typically
  want only one. If `tf.data` is the only thing you need from
  TensorFlow, `tensorflow-cpu` is enough and won't fight with `libtpu`.
- **Forgetting to add a new package locally** — Kinetic only sees what's
  in `requirements.txt` or `pyproject.toml`. A `pip install` in your
  shell that isn't reflected in those files won't carry over.
- **Massive dependency sets** — every `requirements.txt` change forces
  a bundled rebuild. If your deps churn daily, consider prebuilt mode
  (after publishing a base image with `kinetic build-image`).
- **Editable installs (`pip install -e`)** — an editable install does not
  show in `requirements.txt`, and Kinetic cannot carry it over. Keep the
  source inside the [package root](packaging.md), which Kinetic packages
  for you. As an alternative, publish the package and pin a released
  version. `pip` cannot install a line such as `-e .` on the pod. `pip`
  installs a line such as `-e git+https://example.com/pkg` correctly,
  because that line names a remote source.
- **Local path references** — the lines `-r other.txt`,
  `-c constraints.txt`, `./wheels/foo.whl`, `file://...`, and
  `mypkg @ ./vendor` point to paths that do not exist in the image or on
  the pod. In prebuilt mode, Kinetic refuses these lines at submit time
  and names the line. In bundled mode, the install fails inside Cloud
  Build.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1em` What Ships to the Pod
:link: packaging
:link-type: doc

The package root, and why the dependency file can sit outside the
archive.
:::

:::{grid-item-card} {octicon}`zap;1em` Execution Modes
:link: execution_modes
:link-type: doc

Where the discovered deps go.
:::

:::{grid-item-card} {octicon}`stack;1em` Container Images
:link: containers
:link-type: doc

Custom image and base-image workflows.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

What to check when an import fails on the remote.
:::
::::
