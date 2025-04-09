(contributing_pygem_target)=
# PyGEM Contribution Guide

Before contributing to PyGEM, it is recommended that you either clone [PyGEM's GitHub repository](https://github.com/PyGEM-Community/PyGEM) directly, or initiate your own fork (as described [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)) to then clone.

If PyGEM was already installed in your conda environment (as outlined [here](install_pygem_target)), it is recommended that you first uninstall:
```
pip uninstall pygem
```

Next, clone PyGEM. This will place the code at your current directory, so you may wish to navigate to a desired location in your terminal before cloning:
```
git clone https://github.com/PyGEM-Community/PyGEM.git
```
If you opted to create your own fork, clone using appropriate repo URL: `git clone https://github.com/YOUR-USERNAME/PyGEM.git`

Navigate to root project directory:
```
cd PyGEM
```

Install PyGEM in 'editable' mode:
```
pip install -e .
```

Installing a package in editable mode creates a symbolic link to your source code directory (*/path/to/your/PyGEM/clone*), rather than copying the package files into the site-packages directory. This allows you to modify the package code without reinstalling it.<br>

## General
- The `dev` branch is the repository's working branch and should almost always be the base branch for Pull Requests (PRs). Exceptions include hotfixes that need to be pushed to the `master` branch immediately, or updates to the `README`.
- Do not push to other people's branches. Instead create a new branch and open a PR that merges your new branch into the branch you want to modify.

## Issues
- Check whether an issue describing your problem already exists [here](https://github.com/PyGEM-Community/PyGEM/issues).
- Keep issues simple: try to describe only one problem per issue. Open multiple issues or sub-issues when appropriate.
- Label the issue with the appropriate label (e.g., bug, documentation, etc.).
- If you start working on an issue, assign it to yourself. There is no need to ask for permission unless someone is already assigned to it.

## Pull requests (PRs)
- PRs should be submitted [here](https://github.com/PyGEM-Community/PyGEM/pulls).
- PRs should be linked to issues they address (unless it's a minor fix that doesn't warrant a new issue). Think of Issues like a ticketing system.
- PRs should generally address only one issue. This helps PRs stay shorter, which in turn makes the review process easier.
- Concisely describe what your PR does. Avoid repeating what was already said in the issue.
- Assign the PR to yourself.
- First, open a Draft PR. Then consider:
    - Have you finished making changes?
    - Have you added tests for all new functionalities you introduced?
    - Have you run the ruff linter and formatter? See [the linting and formatting section below](ruff_target) on how to do that.
    - Have all tests passed in the CI? (Check the progress in the Checks tab of the PR.)
  
  If the answer to all of the above is "yes", mark the PR as "Ready for review" and request a review from an appropriate reviewer. If in doubt of which reviewer to assign, assign [drounce](https://github.com/drounce).
- You will not be able to merge into `master` and `dev` branches without a reviewer's approval.

### Reviewing PRs and responding to a review
- Reviewers should leave comments on appropriate lines. Then:
  - The original author of the PR should address all comments, specifying what was done and in which commit. For example, a short response like "Fixed in [link to commit]." is often sufficient. 
  - After responding to a reviewer's comment, do not mark it as resolved.
  - Once all comments are addressed, request a new review from the same reviewer. The reviewer should then resolve the comments they are satisfied with.
- After approving someone else's PR, do not merge it. Let the original author of the PR merge it when they are ready, as they might notice necessary last-minute changes.

(ruff_target)=
## Code linting and formatting
PyGEM **requires** all code to be linted and formatted using [ruff](https://docs.astral.sh/ruff/formatter). Ruff enforces a consistent coding style (based on [Black](https://black.readthedocs.io/en/stable/the_black_code_style/index.html)) and helps prevent potential errors, stylistic issues, or deviations from coding standards. The configuration for Ruff can be found in the `pyproject.toml` file.

⚠️ **Both linting and formatting must be completed before code is merged.** These checks are run automatically in the CI pipeline. If any issues are detected, the pipeline will fail.

### Lint the codebase
To lint the codebase using Ruff, run the following command:
```
ruff check /path/to/code
```
Please address all reported errors. Many errors may be automatically and safely fixed by passing `--fix` to the above command. Other errors will need to be manually addressed.

### Format the codebase
To automatically format the codebase using Ruff, run the following command:
```
ruff format /path/to/code
```
