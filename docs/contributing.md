# Contributing

We welcome your patches and contributions to this project. This page
explains the legal requirements, the development setup, and the review
process.

## Before you begin

### Sign our Contributor License Agreement

Every contribution to this project needs a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You, or your employer, keep the copyright to your contribution. The CLA
gives us the permission to use and to redistribute your contribution as
part of the project.

:::{note}
If you or your current employer already signed the Google CLA, for this
project or for a different one, you do not need to sign it again.
:::

Visit <https://cla.developers.google.com/> to see your current agreements
or to sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Development setup

1. Install the package with development dependencies:

   ```bash
   uv pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

### Code quality and testing

Before you open a pull request, make sure that your changes pass the
linter and the unit tests.

- **Lint:** We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run it with:
  ```bash
  ruff check .
  ```
- **Unit tests:** We use [pytest](https://docs.pytest.org/) to run the unit tests. Run them with:

  ```bash
  pytest
  ```

- **End-to-end tests:** These tests run real workloads on a GKE cluster. They live in `tests/e2e/`. pytest skips them unless you set `E2E_TESTS=1`.

  **Prerequisites:**
  - A Google Cloud project with a Kinetic cluster (`kinetic init`).
  - A Google Cloud login (`gcloud auth login` and `gcloud auth application-default login`).
  - `kubectl` credentials for the cluster. `kinetic init` and `kinetic up` configure them.
  - The test dependencies: `uv pip install -e ".[test]"`.

  **Required environment variables:**

  | Variable          | Required | Default         | Description                    |
  | ----------------- | -------- | --------------- | ------------------------------ |
  | `E2E_TESTS`       | Yes      | —               | Set to `1` to enable e2e tests |
  | `KINETIC_PROJECT` | Yes      | —               | Google Cloud project ID        |
  | `KINETIC_ZONE`    | No       | `us-central1-a` | GKE cluster zone               |
  | `KINETIC_CLUSTER` | No       | `kinetic-cluster` | GKE cluster name             |

  **Run all e2e tests:**

  ```bash
  E2E_TESTS=1 KINETIC_PROJECT=my-project python -m pytest tests/e2e/ -v -n auto
  ```

  **Run a specific test file:**

  ```bash
  E2E_TESTS=1 KINETIC_PROJECT=my-project python -m pytest tests/e2e/cpu_execution_test.py -v
  ```

  :::{tip}
  Remove `-n auto` to run the tests one at a time. Serial runs are easier to debug.
  :::

### Submitting changes

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m 'Add my feature'`.
4. Push the branch: `git push origin feature/my-feature`.
5. Open a pull request.

### Code reviews

Every submission needs a review, including a submission from a project
member. We use GitHub pull requests for reviews. See
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for
more information about pull requests.

## Documentation Contribution Process

Install the documentation dependencies, then build and serve the site
locally:

```bash
uv pip install -e ".[docs]"
sphinx-autobuild docs /tmp/docs
```

The pages are MyST Markdown under `docs/`. Follow the style of the
existing pages: short sentences in the active voice, present tense,
imperative steps, and no contractions.

## Releases

Only maintainers release Kinetic. The process is in
[RELEASE_PROCESS.md](https://github.com/keras-team/kinetic/blob/main/RELEASE_PROCESS.md)
in the repository. In short:

:::{container} kinetic-steps
1. **Bump the version** in `pyproject.toml` and `kinetic/version.py`
   through a pull request.
2. **Create a release branch** named after the version, for example
   `r0.0.5`.
3. **Create a GitHub release** from that branch at
   <https://github.com/keras-team/kinetic/releases/new>.
4. **Wait for the publish workflow.** The release tag starts the
   `publish_to_pypi` GitHub Actions workflow, which uploads the package
   to PyPI. Do not upload with `twine` yourself.
:::
