# Profiles

A **profile** is a saved set of the four values that decide where a job
runs: the Google Cloud project, the zone, the cluster, and the Kubernetes
namespace. One profile is **active** at a time. Every `kinetic` command
and every `@kinetic.run()` call reads the active profile. You therefore
set the target one time, and you do not repeat it in code, in commands,
or in environment variables.

`kinetic init` creates your first profile and makes it active. This page
explains how to create more profiles, how to switch between them, and
how a profile combines with the other configuration sources.

## When you need more than one profile

One profile per cluster is the normal pattern. Create a second profile
when:

- You work with two clusters, for example a personal TPU cluster and a
  shared GPU cluster.
- You keep clusters in two zones or two regions, and you switch when one
  zone has no Spot capacity.
- You work in two projects, or in two namespaces of one cluster.
- Two people share one user account on one machine and need separate
  configurations.

## Create a profile

`kinetic init` and `kinetic up` save a profile automatically and make it
active. The profile name defaults to the cluster name. Pass
`--profile-name NAME` to either command to select a different name. If a
profile with that name exists, `kinetic up` overwrites it, because the
cluster that `up` just created is authoritative.

A profile name starts with a letter or a digit, and contains only
letters, digits, `-`, and `_`, up to 64 characters.

To save a profile for a cluster that already exists, run
`kinetic profile create`:

```bash
kinetic profile create dev-tpu \
  --project my-ml-dev --zone us-central2-b --cluster dev-tpu

kinetic profile create team-gpu \
  --project my-ml-prod --zone us-east1-b --cluster team-gpu \
  --namespace alice
```

You can omit any flag. Kinetic then reads the value from the matching
`KINETIC_*` environment variable, and prompts you if the variable is not
set. The first profile that you create becomes active. After that,
`kinetic profile create` does not change the active profile.
`kinetic init`, `kinetic up`, `kinetic profile use`, `kinetic profile
unset`, and `kinetic profile rm` do.

## Switch and inspect

```bash
kinetic profile ls          # lists profiles; * marks the active one
kinetic profile use team-gpu
kinetic profile show        # prints the active profile
kinetic config              # prints each resolved value and its source
```

## Commands

| Command | What it does |
| ------- | ------------ |
| `kinetic profile create [NAME]` | Saves a new profile. Reads a missing field from the `KINETIC_*` environment variable, then prompts. The first profile becomes active. `--force` overwrites a profile with the same name. |
| `kinetic profile ls` | Lists all profiles. `*` marks the active one. |
| `kinetic profile use NAME` | Makes `NAME` the active profile. |
| `kinetic profile unset` | Clears the active profile. Commands then read only environment variables, flags, and defaults. |
| `kinetic profile show [NAME]` | Prints the fields of a profile. Defaults to the active profile. |
| `kinetic profile rm NAME` | Deletes a profile. Prompts unless you pass `--yes`. If you delete the active profile, no other profile becomes active. Run `kinetic profile use` to select one. |

## Use a different profile for one command

Pass `--profile NAME` to the root `kinetic` command, or set
`KINETIC_PROFILE=NAME`. The stored active profile does not change. The
flag belongs to the root command, so it must come **before** the
subcommand:

```bash
# Correct: the flag is on the root command.
kinetic --profile team-gpu jobs list

# Also correct.
KINETIC_PROFILE=team-gpu kinetic jobs list

# Wrong: `--profile` after the subcommand is not the global flag.
kinetic jobs list --profile team-gpu
```

When the selected profile is not the stored active profile,
`kinetic profile ls` marks the selected profile with `*` and prints a
line such as `Active profile: team-gpu (override; stored: dev-tpu)`.

Python code reads `KINETIC_PROFILE` too. To run one script against a
different profile, set the variable for that process:

```bash
KINETIC_PROFILE=team-gpu python train.py
```

## How a profile combines with the other sources

For each of the four values, Kinetic reads these sources in order and
uses the first value that it finds:

```text
decorator argument or CLI flag  >  KINETIC_* env var  >  active profile  >  built-in default
```

Concretely:

- `@kinetic.run(cluster="adhoc")` runs on `adhoc`, whatever the profile
  says.
- `KINETIC_PROJECT=other-proj kinetic up` targets `other-proj`.
- `kinetic status --cluster adhoc` inspects `adhoc`.

Use the profile for the target that you use every day. Use a flag, an
argument, or an environment variable for a one-off override. Run
`kinetic config` to see which source supplied each value. See
[Configuration](../configuration.md) for the full precedence table.

## Where profiles are stored

Profiles are local to your machine and to your user account. Kinetic does
not sync them to teammates or to other machines. To make a workflow
reproducible for others, write down the four values, or the
`kinetic profile create` command that sets them.

Kinetic stores all profiles in one JSON file at
`~/.kinetic/profiles.json`:

```json
{
  "current": "team-gpu",
  "profiles": {
    "dev-tpu":  { "project": "my-ml-dev",  "zone": "us-central2-b", "cluster": "dev-tpu",  "namespace": "default" },
    "team-gpu": { "project": "my-ml-prod", "zone": "us-east1-b",    "cluster": "team-gpu", "namespace": "alice" }
  }
}
```

You can edit the file by hand, but the CLI is the supported path. Kinetic
writes the file atomically, so an interrupted write does not corrupt the
file. To store the file elsewhere, for example in CI, set
`KINETIC_PROFILES_FILE` to a different path.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`gear;1em` Configuration
:link: ../configuration
:link-type: doc

The full precedence table and every `KINETIC_*` variable.
:::

:::{grid-item-card} {octicon}`server;1em` Clusters and Node Pools
:link: clusters
:link-type: doc

Create, share, and delete the clusters that profiles point at.
:::

:::{grid-item-card} {octicon}`terminal;1em` CLI Reference
:link: ../cli
:link-type: doc

Generated reference for every command and flag.
:::
::::
