# Advanced Workflows

Guides for scaling Kinetic across multiple clusters, using capacity reservations, and managing long-running or batched jobs.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em` Detached Jobs
:link: async_jobs
:link-type: doc

Submit with `run_async()`, then poll status, tail logs, and collect
results — even from a different machine.
:::

:::{grid-item-card} {octicon}`stack;1em` Batched Jobs
:link: batched_jobs
:link-type: doc

Run the same function over many inputs with `run_async_map()` and manage
the whole fan-out through one `BatchHandle`.
:::

:::{grid-item-card} {octicon}`server;1em` Multiple Clusters
:link: clusters
:link-type: doc

Run multiple independent clusters in the same GCP project for isolation,
regions, and environments.
:::

:::{grid-item-card} {octicon}`cpu;1em` Capacity Reservations
:link: reservations
:link-type: doc

Guarantee accelerator hardware is available when your node pool scales up.
:::
::::

```{toctree}
:hidden:

async_jobs
batched_jobs
clusters
reservations
```
