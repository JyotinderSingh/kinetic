# Dependencies

Kinetic reads one dependency file for each job: a `requirements.txt` or a
`pyproject.toml`. Kinetic installs the packages in that file, and your
function can import them on the pod. This page explains how Kinetic finds
the file and which lines Kinetic filters. It also explains how to install
packages from a private index, and lists the pitfalls that cause a missing
import on the pod. The container image mode decides where the packages
install: in the image at build time, or in the pod at start. See
[Container Images](containers.md) for that choice.

## A first run

Put a `requirements.txt` next to your script. Kinetic finds the file
without configuration:

```text
# requirements.txt
keras
numpy
pandas
```

```python
@kinetic.run(accelerator="tpu-v5litepod-4")
def train():
  import pandas as pd  # installed on the pod

  ...
```

A `pyproject.toml` works too. Kinetic reads the `[project.dependencies]`
list from that file. If both files are in the same directory, Kinetic
uses `requirements.txt`.

:::{tip}
**Recommended defaults:**

- List only the packages that your function imports. A short list makes
  the image build faster.
- Do not pin `jax`, `jaxlib`, `libtpu`, or `libtpu-nightly`. Kinetic
  filters those lines and installs the JAX version that matches the
  accelerator.
- If you already have a `pyproject.toml` for local development, use that
  file. You do not need a separate `requirements.txt`.
:::

## How discovery works

When you call a decorated function, Kinetic starts at the
[entry directory](packaging.md#the-package-root): the directory of the
module file that defines the function. Kinetic then walks **up** one
directory at a time. At each directory, Kinetic does these steps:

1. If a file with the name `requirements.txt` is in the directory,
   Kinetic uses that file.
2. If not, and a file with the name `pyproject.toml` is in the directory,
   Kinetic uses that file and reads `[project.dependencies]` from it.
3. If neither file is in the directory, Kinetic moves up one directory
   and does the steps again.

Kinetic examines files only. If a directory has the name
`requirements.txt` or `pyproject.toml`, Kinetic ignores that directory
and continues the walk.

The walk has bounds, so that Kinetic does not use a file from outside
your project:

- Kinetic stops after it examines the first directory that holds a `.git`
  entry. The entry can be a directory or a file, because a git worktree
  uses a file. Kinetic examines that directory for a dependency file
  before the walk stops.
- Kinetic stops at your home directory and at the root of the file
  system. Kinetic examines these directories too.

If Kinetic finds no dependency file, the pod gets only the packages that
Kinetic installs in every image: JAX, Keras, `cloudpickle`,
`google-cloud-storage`, and Kinetic.

Kinetic writes the name of the selected file into the log on each submit:
`Using dependency file: ...`. If the installed packages are not the
packages that you expected, read this log line first. If both files are
in the selected directory, the log also names the file that Kinetic
selected. Kinetic logs a warning when it finds no `.git` entry and the
selected file is not in your entry directory. The log therefore always
shows when Kinetic uses a file from outside your entry directory.

:::{warning}
A `pyproject.toml` file with no `[project.dependencies]` list still stops
the walk. If your dependencies are in a `requirements.txt` file above
that directory, Kinetic does not find them. Kinetic logs a warning that
the selected file declares no dependencies, and that the pod gets only
the packages that Kinetic installs in every image. Move or copy your
dependency file into your entry directory.

Kinetic reads `[project.dependencies]` only. If that list is empty or
absent, Kinetic looks for dependencies in `[tool.poetry.dependencies]`,
`[project.optional-dependencies]`, `[dependency-groups]`, and
`dynamic = ["dependencies"]`. If Kinetic finds any of those, Kinetic logs
a warning that names the tables. If `[project.dependencies]` is not
empty, Kinetic installs that list and ignores the other tables without a
warning.
:::

Kinetic ships only the [package root](packaging.md) in `context.zip`.
Kinetic can therefore select a dependency file above the package root.
Kinetic installs the packages from that file, but the directory of that
file is not on the pod.

## What Kinetic does with the file

Kinetic logs the selected file. Kinetic then filters the content (see
[JAX and accelerator runtimes](#jax-and-accelerator-runtimes)), hashes
the filtered content into the image tag, and writes the filtered content
to a generated `requirements.txt` for Cloud Build. A change to the
filtered content causes a new image build on the next run. The build
sees the generated file alone. A line that points to a path on your
machine (`-r base.txt`, `-e .`, or `./local-wheel`) therefore cannot
resolve, and the install fails inside Cloud Build.

The other [container image modes](containers.md) change where the
install happens. The prebuilt mode installs a generated file on the pod
at start, and rejects a local path line at submit time. A custom image
ignores the dependency file.

## JAX and accelerator runtimes

The image that Kinetic builds already contains `jax`, `jaxlib`, and the
runtime for the accelerator category: `libtpu` on TPU, or the CUDA
libraries on GPU. To prevent your dependency file from replacing that
installation, Kinetic removes these entries before the install:

- `jax`
- `jaxlib`
- `libtpu`
- `libtpu-nightly`

Kinetic logs a warning for each removed line. The warning names the
package and tells you how to keep the line.

If you must override the JAX version, for example to test a new release,
append `# kn:keep` to the line:

```text
jax==0.4.25 # kn:keep
jaxlib==0.4.25 # kn:keep
```

The marker works in `requirements.txt`. Use the marker with care. A
mismatch between `jax`, `jaxlib`, and the accelerator runtime is a common
cause of crashes that are hard to diagnose.

:::{note}
Kinetic filters physical lines for the image build. If a filtered `jax`
entry continues onto more lines with a backslash, the continuation lines
stay in the file and can break the install. The output of
`pip-compile --generate-hashes` has such lines. Keep each entry on one
line.
:::

## Private packages

Cloud Build installs your dependencies into the image. Cloud Build
never receives your project source. The build context holds exactly three
things: the generated Dockerfile, the Kinetic runner script, and the
generated `requirements.txt`. A `pip.conf`, an environment variable, or a
credential on your machine or in your project tree does not reach the
build. The only way to tell the installer about a private index is a line
in the dependency file itself.

You have two options:

::::{tab-set}

:::{tab-item} An index URL in requirements.txt

Add `--index-url` or `--extra-index-url` as a line in `requirements.txt`.
The installer reads these directives and uses them for every package in
the file:

```text
--extra-index-url https://my-org-private-index.example.com/simple
my-private-package==1.2.3
some-public-dep==2.0.0
```

This option does not work in a `pyproject.toml`, because `[project.dependencies]` holds
package specifiers only. This option needs an index that requires no
credentials. Examples: a public index, or an index that the Cloud Build
worker and the pod can reach through network rules alone.
:::

:::{tab-item} A custom image

If your private packages need credentials at install time, system
libraries, or special build flags, build a container image that contains
those packages. Pass the image as `container_image="<your-image-uri>"`.
You control the build environment: `pip.conf`, secret mounts, and
`gcloud` authentication. See
[Container Images](containers.md#custom-image-mode).
:::

::::

:::{warning}
Do not put a secret into `requirements.txt`, for example
`https://user:token@host/...`. Kinetic uploads the generated
requirements file to the builds bucket as part of the Cloud Build source
(or to the jobs bucket in the prebuilt mode). Anyone with read access to
those buckets or to the build can read the token.
:::

## Common dependency pitfalls

- **A `jax` pin without `# kn:keep`.** Kinetic drops the line, logs a
  warning, and installs the JAX version of the image. If you want the
  pin, add `# kn:keep`. If you do not want the pin, delete the line.
- **TensorFlow next to JAX.** The `tensorflow` package can also try to
  use the TPU, and JAX then cannot open the TPU. If you need TensorFlow
  for `tf.data` only, install `tensorflow-cpu`. That package does not use
  the TPU.
- **A package that you installed locally but did not list.** Kinetic
  reads `requirements.txt` or `pyproject.toml` only. A package that you
  installed with `pip install` in your shell, but did not list in one of
  those files, is not on the pod.
- **Extras in `pyproject.toml`.** Kinetic reads `[project.dependencies]`
  only. Packages in `[project.optional-dependencies]` or in
  `[dependency-groups]` do not install, and Kinetic logs no warning when
  `[project.dependencies]` is not empty. Move those packages into
  `[project.dependencies]`, or use a `requirements.txt`.
- **A large dependency set that changes often.** A change to the
  dependency file causes a new image build. If your dependencies change
  many times a day, see [Container Images](containers.md) for a mode that
  installs at pod start.
- **An editable install (`pip install -e`).** An editable install does not
  appear in `requirements.txt`, and Kinetic cannot install it on the pod.
  Keep the source inside the [package root](packaging.md), which Kinetic
  ships for you. As an alternative, publish the package and pin a released
  version. A line such as `-e .` cannot install in the image or on the
  pod. Kinetic accepts a line such as `-e git+https://example.com/pkg`,
  because that line names a remote source.
- **A local path reference.** Examples: `-r other.txt`,
  `-c constraints.txt`, `./wheels/foo.whl`, `file://...`, and
  `mypkg @ ./vendor`. These lines point to paths that do not exist in the
  image or on the pod. The install fails inside Cloud Build. The prebuilt
  mode rejects these lines at submit time and names the line.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1em` What Ships to the Pod
:link: packaging
:link-type: doc

The package root, and why the dependency file can sit outside the
archive.
:::

:::{grid-item-card} {octicon}`stack;1em` Container Images
:link: containers
:link-type: doc

Where the packages install: in the image at build time, in the pod at
start, or in an image that you build.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

What to check when an import fails on the pod.
:::
::::
