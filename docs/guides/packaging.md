# What Ships to the Pod

This page is the exact contract for the artifacts that Kinetic sends to
the pod. It describes what goes into them, where the pod stores them, and
how the pod rebuilds your project. Read this page when you debug a
`ModuleNotFoundError`, a `FileNotFoundError`, or a job that uploads
gigabytes. Then read [Troubleshooting](../troubleshooting.md).

The pod downloads two artifacts from Cloud Storage for every job:

1. **`payload.pkl`** — your function, the objects that it closes over,
   its arguments, the environment variables that you capture, and a
   fingerprint of your client toolchain. Kinetic serializes all of these
   with `cloudpickle`.
2. **`context.zip`** — a snapshot of your project source, rooted at the
   *package root*. Kinetic writes the packaging plan into the same
   archive, at the reserved path `.kinetic/plan.json`.

Kinetic also reads one dependency file: a `requirements.txt` or a
`pyproject.toml`. That file decides which Python packages the pod
provides. See [Dependencies](dependencies.md).

## The package root

The package root is the directory that Kinetic archives. Kinetic resolves
the package root at submit time, in three steps.

**Step 1 — find the entry directory.** Kinetic finds the module that
defines the decorated function, and takes the directory of that module
file. A notebook cell, a REPL, and `python -c` define no module file. In
those cases Kinetic uses your current working directory instead.

**Step 2 — escape the package.** Kinetic walks up from the entry
directory for as long as each directory holds an `__init__.py` file.
This step makes `from trainer.model import ...` resolvable on the pod,
because the root ends above `trainer/`, and not inside it. This walk
stops at your home directory and at the root of the file system.

**Step 3 — walk up to a project marker.** Kinetic then walks up to the
nearest directory that holds one of these markers:

- `pyproject.toml`
- `requirements.txt`
- `setup.py`
- `setup.cfg`
- `.git` (a directory, or a file — a git worktree uses a file)

The walk stops at your home directory and at the root of the file system.
Kinetic never adopts either one as the package root, unless step 2 already
ended there. If Kinetic finds no marker, Kinetic keeps the directory from
step 2.

**Override.** Set `KINETIC_PACKAGE_ROOT` to select the root yourself:

```bash
export KINETIC_PACKAGE_ROOT=/home/me/monorepo/services/trainer
```

The override replaces all detection. Kinetic expands a leading `~` in the
value. Kinetic then validates the value, and raises a `ValueError` at
submit time in two cases:

- The value does not name an existing directory.
- The value is neither the entry directory nor a parent of it.

:::{tip}
Two rules keep the root predictable:

- Keep a `pyproject.toml`, a `requirements.txt`, or a git repository at
  the top of the tree that you want to ship.
- Keep large data out of that tree. As an alternative, wrap the data in
  `kinetic.Data(...)`. Kinetic excludes such a path from `context.zip`
  automatically.
:::

### Worked examples

| Layout | Decorated function in | Package root | Why |
| ------ | --------------------- | ------------ | --- |
| `proj/train.py` (+ `proj/requirements.txt`) | `train.py` | `proj/` | The entry directory already holds a marker. |
| `proj/trainer/model.py` (+ `proj/pyproject.toml`, `trainer/__init__.py`) | `model.py` | `proj/` | Step 2 escapes the package. Step 3 finds the marker. |
| `proj/src/pkg/train.py` (+ `pkg/__init__.py`, `proj/pyproject.toml`) | `train.py` | `proj/` | Kinetic archives `src/` too, so the plan can add it to `sys.path`. |
| `~/scratch/one_off.py`, no markers anywhere | `one_off.py` | `~/scratch/` | No marker exists. Kinetic keeps the entry directory, and never `$HOME`. |
| A Jupyter notebook in `~/nb/`, no markers anywhere | the notebook cell | `~/nb/` | The cell has no `__file__`. Kinetic uses the current directory. |

## What Kinetic excludes

Kinetic always excludes these two directory names, at any depth:

`.git`, `__pycache__`

This rule applies to directory names only. In a git worktree, `.git` is
a small file, and Kinetic archives that file.

Kinetic also excludes these names at any depth, unless you turn the
default exclusions off:

`.venv`, `venv`, `node_modules`, `.tox`, `.mypy_cache`, `.ruff_cache`,
`.pytest_cache`, `.ipynb_checkpoints`, `.DS_Store`

Kinetic also excludes a local path that you wrap in `kinetic.Data(...)`,
if that path lies inside the package root. Kinetic uploads that data
through the content-addressed data cache instead, and does not upload the
data a second time. See [Working with Data](data.md).

To turn the default exclusions off, set this variable:

```bash
export KINETIC_NO_DEFAULT_EXCLUDES=1
```

The value must be exactly `1`. `.git` and `__pycache__` stay excluded.

### `.kineticignore`

Put a `.kineticignore` file at the package root to exclude more paths.
Kinetic reads this file at the package root only. Write one pattern per
line, in `fnmatch` syntax. A line that starts with `#` is a comment.
Kinetic reads a `#` in any other position as part of the pattern. A
trailing `/` restricts the pattern to directories. Kinetic removes a
leading `/`, so `/build/` and `build/` behave the same.

```text
# .kineticignore
checkpoints/
*.ckpt
scratch_*.py
docs/_build/
```

Kinetic matches every pattern two times:

- Against the path relative to the package root, such as `docs/_build`.
- Against the last name in that path, such as `_build`.

The second match applies at any depth. `*.ckpt` therefore excludes a
checkpoint file anywhere in the tree, and `checkpoints/` excludes every
directory with that name.

### Secret files

Kinetic does not exclude secret files by default, because some projects
need them on the pod. Kinetic logs a warning instead when a file name in
the archive matches one of these patterns:

- `.env*`
- `*.pem`
- `id_rsa*`

The warning names every matched file. If you did not intend to send one
of the files, do these steps:

1. Add the file to `.kineticignore`.
2. Submit the job again.

### Size warnings

Kinetic logs the archive size on every submission. Kinetic logs a warning
above 100 MB, and lists the five largest files in the archive. Change the
threshold with `KINETIC_CONTEXT_SIZE_WARN_MB`.

The pickled payload has a separate threshold of 50 MB
(`KINETIC_PAYLOAD_SIZE_WARN_MB`). A large payload usually means one
module-level global that Kinetic captured **by value**. `cloudpickle`
serializes the objects that your function references. A module-level
`DF = pd.read_parquet(...)` that your function reads therefore goes
into `payload.pkl` on every submission. Pass large data as
`kinetic.Data(...)`, or load it inside the function.

Set either threshold to `0` to turn that warning off. Kinetic ignores a
value that is not a number, and logs a warning about the bad value.

### Fidelity of the archive

`context.zip` is an exact snapshot, with these rules:

- Kinetic follows a symlinked directory, and guards against a cycle.
  Kinetic archives each real directory one time only, and names every
  skipped directory in a warning.
- Kinetic skips a broken symlink and an unreadable file, and logs a
  warning for each one. One bad file never stops a submission.
- Kinetic stores the POSIX mode of each file. The runner restores the
  read, write, and execute bits on the pod, for an archive that a Unix
  client built. The runner never restores a setuid, setgid, or sticky
  bit.
- Kinetic archives an empty directory, and the runner restores it.
- Kinetic clamps a timestamp that falls outside the range of the ZIP
  format, which is the year 1980 to the year 2107.

## What happens on the pod

The pod downloads both artifacts and verifies their SHA-256 hashes. The
runner then does these steps:

:::{container} kinetic-steps
1. The runner extracts `context.zip` into the workspace directory.
2. The runner rebuilds `sys.path` from the packaging plan. The workspace
   root goes first. Each client `sys.path` entry that lived under the
   package root follows it, at the same relative path inside the
   workspace. This step makes a `src/` layout work without a
   `PYTHONPATH` on the pod.
3. The runner changes the working directory to the workspace directory
   that matches your client working directory. The runner uses the
   workspace root when your client working directory was outside the
   package root.
4. The runner unpickles `payload.pkl`.
5. The runner applies the captured environment variables and resolves
   the `Data` references.
6. The runner calls your function.
:::

The runner does not replicate a client `sys.path` entry that points at
site-packages, or at any directory outside the package root. The pod
environment provides those packages.

Step 3 has one practical consequence: **a relative path behaves the same
on the pod as it does on your machine**. `open("configs/train.yaml")`
works remotely if it works locally from the same directory. The file
must be inside the package root, and it must not be excluded.

An absolute client path is not a supported way to read a shipped file.
The runner does create one symbolic link, at the path of your client
**entry directory**. The entry directory is the directory of the module
file that defines the decorated function. The link points at the
workspace root, and the runner creates it only when that path does not
exist on the pod. The link exists so that the debugger can map source
files. When the entry directory sits below the package root, the link
still points at the workspace root, and not at the matching
subdirectory. Use relative paths for portable access to files.

The pod environment also holds `KINETIC_OUTPUT_DIR`. The value is a Cloud
Storage URI, such as `gs://{project}-kn-{cluster}-jobs/outputs/{job_id}`.
The pod filesystem, including the workspace, is gone when the pod ends.
Write everything that you want to keep to that Cloud Storage location.
See [Outputs and Checkpoints](checkpointing.md).

## How imports resolve remotely

Kinetic uses three mechanisms. The mechanism that applies to a module
explains almost every remote import failure.

**Kinetic ships your first-party code by value.** At submit time Kinetic
inspects `sys.modules`. Kinetic registers each module whose file lives
under the package root with `cloudpickle.register_pickle_by_value`.
`cloudpickle` then serializes the functions and classes of those modules
into `payload.pkl` in full. The pod therefore does not import your
modules to unpickle your job. A helper in `trainer/utils.py` travels
inside the payload. The pod does not run the module-level statements of
those modules, because the pod does not import them.

Kinetic never registers these two modules by value:

- The `kinetic` package itself, even when you work inside a checkout of
  Kinetic.
- The `__main__` module. `cloudpickle` already ships a `__main__`
  function by value.

**The pod resolves third-party imports.** `cloudpickle` pickles anything
outside the package root by reference: `numpy`, `keras`, and your private
packages. The pod environment must provide those packages, and that is
the purpose of the dependency file.

**The workspace resolves runtime imports.** An `import` statement inside
your function runs at call time, against the pod `sys.path`. That path
starts with the workspace. The pod finds a first-party module that you
import lazily in the extracted context. The pod environment must provide
a third-party package that you import lazily.

:::{note}
**Identity limit of shipping by value.** A class that travels by value
is a different type object from the same class that the pod imports. An
`isinstance` check across the two paths can return `False`. Compare by
attribute, or by name, when you must cross that boundary.
:::

:::{note}
**A decorated function inside the payload.** The payload can reference a
second `@kinetic.run()` function, for example as an attribute of an
object that you pass. That function arrives on the pod as the plain,
undecorated function. A call to it on the pod runs the body in the pod
process. Kinetic does not support a nested job submission.
:::

## Arguments: which types Kinetic preserves

Kinetic pickles the arguments and the keyword arguments with the
function. Kinetic walks them for one purpose only: to replace each
`Data(...)` object with a reference. Kinetic scans the arguments on every
submit. Kinetic rebuilds the containers only when the call holds a `Data`
object. The pod also skips its own walk when the payload holds no
reference.

The walk keeps types and object identity:

- Kinetic rebuilds a `list`, and a list subclass, as the same type.
- Kinetic rebuilds a `tuple` and a `NamedTuple` as the same type, with
  the fields intact. Both `typing.NamedTuple` and
  `collections.namedtuple` work.
- Kinetic passes a `set` and a `frozenset` through unchanged when no
  `Data` object is reachable inside it. If a `Data` object is reachable,
  Kinetic rejects the argument (see below).
- Kinetic preserves a `dict` subclass. `OrderedDict` keeps its order,
  `defaultdict` keeps its `default_factory`, and `Counter` keeps its
  type.
- Kinetic preserves aliasing. If one object appears two times in your
  arguments, your function receives the same object two times
  (`out[0] is out[1]`).
- Kinetic pickles everything else whole, and it arrives unchanged.

If Kinetic cannot rebuild a container subclass, Kinetic uses the plain
built-in type instead, and logs a warning. Kinetic does not fail the job.

**Kinetic rejects three argument shapes at submit time.** Each message
names the position of the argument at fault:

- A `Data` object inside a `set` or a `frozenset`, at any depth (for
  example, inside a tuple that sits in a set). The replacement reference
  is a dict, and a dict is not hashable.
- A `Data` object used as a `dict` key.
- A self-referential structure whose cycle runs through a tuple, a set,
  or a frozenset. A cycle through a list or a dict arrives on the pod
  unchanged. Kinetic reports the cycle only when the call also holds a
  `Data` object, because only then does Kinetic rebuild the containers.

Kinetic does not find a `Data` object inside the attributes of a custom
object, because Kinetic walks plain containers only. Kinetic uploads
nothing for that object, and the pod hands your function the original
`Data` instance, which points at a path on your machine. Pass a `Data`
object as its own argument, or inside a list, a tuple, or a dict.

When pickling fails, Kinetic bisects the payload and names the component
at fault. One example message is
`kinetic could not serialize argument 2 (type socket): ...`. Kinetic
counts positional arguments from `0`, and identifies a keyword argument
by name. You therefore never get a bare `PicklingError` from inside
`cloudpickle`.

## Environment variables

`capture_env_vars` sends local environment variables to the pod. Kinetic
always accepts an exact name. A name that ends with `*` is a prefix
pattern: `"WANDB_*"` matches every name that starts with `WANDB_`, and
`"*"` matches every name. A `*` in any other position is not a wildcard.

A prefix pattern never matches the names in this blocklist. The pod
applies your values over its own environment, and these names break the
pod runtime:

`PATH`, `HOME`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `LD_PRELOAD`,
`VIRTUAL_ENV`, `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, `SHELL`, `TMPDIR`,
`TEMP`, `TMP`, `HOSTNAME`, `USER`, `LOGNAME`, `SSH_AUTH_SOCK`,
`KUBERNETES_SERVICE_HOST`, `KERAS_BACKEND`

You can still forward any of these names. List the name exactly:
`capture_env_vars=["KERAS_BACKEND"]` works, and a prefix pattern that
covers it does not.

Kinetic logs the names that it captured, and never the values. Kinetic
logs one more warning when a captured name contains `TOKEN`, `SECRET`,
`KEY`, `PASSWORD`, or `CREDENTIAL`, in any letter case. Kinetic stores
those values inside `payload.pkl` in the jobs bucket, and every job pod
in the cluster can read that bucket. This warning is informational, and
it appears even for a name that you listed exactly. If you intend to send
the credential, you need no action. See
[Forward Environment Variables](env_vars.md) and
[Security](../security.md).

## Notebooks and REPLs

A function defined in a Jupyter cell, an IPython session, or `python -c`
has no source file. Kinetic detects this condition. Kinetic then uses your
current working directory as the entry directory, and logs the directory
that it chose. Steps 2 and 3 then run as usual, so a project marker above
your current directory can move the root further up the tree. Change into
your project directory before you submit.

A notebook function has no importable module, so `cloudpickle` always
pickles it by value. That behavior is correct for a notebook. A helper
from another cell travels with the function. A helper in a `.py` file
beside the notebook travels in `context.zip`.

## Matching your local environment to the pod

Pickled code objects are **not** portable across Python minor versions. A
function that you pickle on Python 3.12 can fail to unpickle on Python
3.11, or can crash the interpreter later. The pod Python must therefore
match your client at the minor version (`X.Y`).

- **The image that Kinetic builds** (the default) starts from
  `python:{X.Y}-slim`, where `X.Y` is the minor version of your local
  interpreter. The build installs `keras`, `cloudpickle`,
  `google-cloud-storage`, JAX for your accelerator category, and
  `keras-kinetic` pinned to your client version. The pod Python
  therefore always matches your client, and you take no action.
- **A base image that you publish** with `kinetic build-image` must have
  the Python minor version of the client that submits the job. Kinetic
  requests the image tag that matches your client Kinetic version.
- **A custom image** makes you responsible for the Python version.

See [Container Images](containers.md) for the three ways to produce the
image and for the requirements of a custom image.

Every payload carries a fingerprint of the client: the Python version,
the `cloudpickle` version, and the Kinetic version. The runner compares
the fingerprint against the pod, and logs a warning for a mismatch. The
runner compares the Python version at the minor version only (`X.Y`), so
`3.12.2` and `3.12.7` produce no warning. The runner compares the
`cloudpickle` version exactly. When the payload does not unpickle, the
error that `result()` raises names both sides, such as
`client Python 3.12.2 / pod Python 3.11.9`. The pod log holds a skew
warning only when the payload unpickled correctly.

## Settings reference

These settings are environment variables on the machine that submits the
job. A profile does not store them.

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `KINETIC_PACKAGE_ROOT` | _(unset)_ | Set the package root. The value must name an existing directory, and must be the entry directory or a parent of it. |
| `KINETIC_NO_DEFAULT_EXCLUDES` | _(unset)_ | Set it to `1` to ship `.venv/`, `node_modules/`, and the rest of the default exclusion list. `.git` and `__pycache__` stay excluded. |
| `KINETIC_CONTEXT_SIZE_WARN_MB` | `100` | The `context.zip` size above which Kinetic logs a warning and lists the largest files. Set it to `0` to turn the warning off. |
| `KINETIC_PAYLOAD_SIZE_WARN_MB` | `50` | The `payload.pkl` size above which Kinetic logs a warning about capture by value. Set it to `0` to turn the warning off. |

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1em` Dependencies
:link: dependencies
:link-type: doc

How Kinetic finds the dependency file, and what the image contains.
:::

:::{grid-item-card} {octicon}`database;1em` Working with Data
:link: data
:link-type: doc

How to move large inputs without the archive.
:::

:::{grid-item-card} {octicon}`container;1em` Container Images
:link: containers
:link-type: doc

How Kinetic produces the image, and the three modes.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

Diagnosis of packaging failures, by symptom.
:::
::::
