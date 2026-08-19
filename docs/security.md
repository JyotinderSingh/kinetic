# Security and Threat Model

Kinetic runs user-defined Python code on remote infrastructure in your
Google Cloud project. Its purpose is arbitrary code execution. This page
states the security boundary, what Kinetic protects against, and what it
does not.

## The security boundary

:::{important}
**Kinetic treats every user who can submit jobs as a trusted user.** A
user who has the IAM permissions to write to the jobs bucket and the
Kubernetes permissions to create Jobs and Pods on the cluster can run
arbitrary code on the cluster.
:::

Kinetic does not sandbox, restrict, or monitor the Python code of an
authorized user. Protection of the cluster against its authorized users
belongs at the infrastructure level: network policies, minimal IAM roles
for the GKE nodes, and namespace isolation.

## What Kinetic protects against

1. **Payload tampering in transit or at rest.** Kinetic serializes your
   function with `cloudpickle`. Deserialization of untrusted pickle data
   is dangerous (CWE-502). An attacker who has write access to the jobs
   bucket but no Kubernetes access could replace the payload.
   *Mitigation:* the client computes a SHA-256 hash of `payload.pkl` and
   of `context.zip` at submit time and writes both hashes into the pod
   specification. The runner verifies both files against those hashes
   before it deserializes anything. *Limit:* the runner does not verify
   the other objects that it downloads from the bucket: the generated
   `requirements.txt` in prebuilt mode, and the `Data` content in the
   data cache. Restrict write access to the jobs bucket to the people
   who can submit jobs.
2. **Data exfiltration through the buckets.** Kinetic creates
   cluster-scoped Cloud Storage buckets. *Mitigation:* the buckets use
   uniform bucket-level access, and only the cluster service accounts get
   IAM bindings. Grant access to developers with the same care.

## What Kinetic does not protect against

1. **Malicious insiders.** Kinetic does not prevent an authorized user
   from writing malicious code inside a `@kinetic.run()` function.
2. **Container escapes.** A container runtime vulnerability is a GKE
   concern, not a Kinetic concern.
3. **Compromised credentials.** An attacker who has a user's `kubeconfig`
   or Google Cloud credentials inherits that user's ability to run code.

## Secrets in jobs

Values that you forward with `capture_env_vars` travel inside
`payload.pkl` in the jobs bucket. Every job pod in the cluster can read
that bucket. Forward only what the job needs, and prefer short-lived
tokens. Kinetic warns when a captured name looks like a credential. See
[Forward Environment Variables](guides/env_vars.md).

Kinetic does not exclude secret files (`.env*`, `*.pem`, `id_rsa*`) from
the source archive by default. Kinetic warns when the archive contains
one. Add the file to `.kineticignore` if you did not intend to ship it.
See [What Ships to the Pod](guides/packaging.md).
