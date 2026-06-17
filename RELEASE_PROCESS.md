# Kinetic Release Process - To be used only by Kinetic maintainers.

Follow these steps to release a new version of Kinetic:

1. **Bump Version:** Create a PR to bump the version number in `pyproject.toml` and `kinetic/version.py` 
   - Example PR: [https://github.com/keras-team/kinetic/pull/270](https://github.com/keras-team/kinetic/pull/270)
2. **Create Release Branch:** Create a new release branch with the release version name.
   - Example branch name: `r0.0.5`
3. **Create a New Release:** Go to [https://github.com/keras-team/kinetic/releases/new](https://github.com/keras-team/kinetic/releases/new) and create a new release.
   <img width="1336" height="1496" alt="image" src="https://github.com/user-attachments/assets/e0c1aaa7-62f8-4893-ba0b-2090fa54b586" />
4. **Publish to PyPI:** Cutting the release will automatically trigger the release to PyPI.
