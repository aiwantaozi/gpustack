# Testing Unreleased Code on a Kubernetes Cluster

This guide covers how to get **uncommitted, unreleased** code from your local working trees —
`gpustack`, `gpustack-runtime` and `gpustack-operator` — into a real Kubernetes cluster so you can
run an end-to-end verification against it.

For the plain local development loop (no cluster), see [Development Guide](./development.md).
For how a cluster gets registered in the first place, see
[Adding a GPU Cluster Using Kubernetes](./tutorials/adding-gpucluster-using-kubernetes.md).

!!! warning "Almost every failure mode here is silent"
    The dangerous part of this workflow is not that it breaks — it's that it *appears* to work
    while running the old code. Two mechanisms cause this (a same-name image tag under
    `imagePullPolicy: IfNotPresent`, and a pod recreation that discards an `exec`-time mutation).
    Both are covered below, and both end with an explicit *verify* step. Do not skip those.

## Reference environment

The commands below were written against this shape. Adapt names, not structure.

| Piece | What it is |
| --- | --- |
| Cluster | Single-node k3s |
| Namespace | `gpustack-system` |
| Worker | DaemonSet `gpustack-worker-nvidia` (`hostNetwork: true`), container name `gpustack-worker` |
| Operator | Deployment `gpustack-operator-worker`, container name `main`, args `gpustack-operator worker -v=2`, binary at `/usr/bin/gpustack-operator` |
| Scheduling | Kueue installed |
| Server | Runs **on the host**, not in the cluster, with `enable_worker: false` |

On k3s, `kubectl` usually needs `sudo k3s kubectl`, or
`export KUBECONFIG=/etc/rancher/k3s/k3s.yaml`. The examples below just say `kubectl`.

## Three components, three different paths

There is no single mechanism that covers all three. Pick the row you need.

| Component | Where it runs | How your local code gets in |
| --- | --- | --- |
| `gpustack` server | Host process | Run it straight from the working tree |
| `gpustack` worker | Container | Build & push an image, **or** overlay the tree via `hostPath` + `PYTHONPATH` |
| `gpustack-operator` | Container (Go) | Build & push an image, **or** overlay the compiled binary via `hostPath` |

The operator is a Go binary. It cannot be injected into a Python environment — no `PYTHONPATH`
trick applies to it.

## 1. Server — run it from the working tree

The server is a host process, so there is nothing to inject: point it at your config and run.

```bash
cd /path/to/gpustack
.venv/bin/python -m gpustack.main start --config-file /path/to/config.yaml
```

Keep credentials out of the config you commit — `database_url`, `bootstrap_password` and any
registry credentials belong in a local, git-ignored file.

### Using a local `gpustack-runtime` too

`gpustack` depends on a pinned `gpustack-runtime` release (see `pyproject.toml`). To run the
server against your own runtime working tree, install it editable:

```bash
uv pip install --python .venv/bin/python -e /path/to/runtime
```

Two things to know:

- **The `.venv` is uv-managed and has no `pip`.** `.venv/bin/pip` does not exist and
  `.venv/bin/python -m pip` fails with `No module named pip`. Use `uv pip` with `--python`, as
  above.
- **`make install` reverts it.** `hack/install.sh` runs `uv sync --locked`, which restores the
  pinned `gpustack-runtime` from the lockfile and drops your editable install. Re-run the
  `uv pip install -e` after any `make install`.

Verify which copy is loaded:

```bash
.venv/bin/python -c "import gpustack_runtime; print(gpustack_runtime.__file__)"
```

### Stopping the server — use a pidfile, not `pgrep -f`

Do **not** do this:

```bash
# WRONG — can kill your own shell
pkill -f "gpustack.main start"
```

`pgrep -f` / `pkill -f` match against the full command line of *every* process, including the
shell that is running the command — that shell's own command line contains the pattern string.
The match set therefore includes your terminal and any wrapper process, and the `kill` takes them
down with the server.

Record the pid instead:

```bash
.venv/bin/python -m gpustack.main start --config-file /path/to/config.yaml &
echo $! > /tmp/gpustack-server.pid

# later
kill "$(cat /tmp/gpustack-server.pid)"
```

## 2. Worker — option A: build and push an image

This is the highest-fidelity path: the pod runs exactly the artifact a release would.

```bash
cd /path/to/gpustack
make package                                        # -> gpustack/gpustack:dev
docker tag gpustack/gpustack:dev <your-registry>/gpustack:<tag>
docker push <your-registry>/gpustack:<tag>
kubectl -n gpustack-system rollout restart ds/gpustack-worker-nvidia
```

`make package` defaults to `gpustack/gpustack:dev`; override with `PACKAGE_NAMESPACE`,
`PACKAGE_REPOSITORY` and `PACKAGE_TAG` (see `hack/package.sh`).

### Prerequisite: `imagePullPolicy: Always`

**If you reuse a tag name you have already deployed, the DaemonSet must be on
`imagePullPolicy: Always`, or the restart will silently run the old code.**

The generated worker manifest hardcodes `imagePullPolicy: IfNotPresent`
(`gpustack/k8s/daemonset.jinja`). Under `IfNotPresent` the kubelet only consults the tag when the
image is *absent* from the node. Push a new image under a tag the node already has, restart the
DaemonSet, and the kubelet resolves the tag locally, finds a copy, and starts the **old** layers.
Nothing errors. The pod is `Running`, the rollout reports success, and your verification run
exercises code you did not write.

Two ways out:

- **Flip the policy** (do this once, then reuse a tag freely):

  ```bash
  kubectl -n gpustack-system patch ds/gpustack-worker-nvidia --type=strategic -p '
  spec:
    template:
      spec:
        containers:
          - name: gpustack-worker
            imagePullPolicy: Always
  '
  ```

- **Never reuse a tag.** Use `pd-1`, `pd-2`, … and patch the image on each push. This works under
  `IfNotPresent` because a new tag is genuinely absent from the node.

The worker DaemonSet is *not* reconciled by the server — the cluster-registration endpoint renders
`manifest.yaml` and hands it to you as a download (`gpustack/routes/clusters.py`). A `kubectl patch`
therefore sticks until someone re-applies a freshly rendered manifest, which will bring back
`IfNotPresent`.

### Verify the pod actually took the new image

Compare the resolved digest before and after, not the tag:

```bash
kubectl -n gpustack-system get pods -l app=gpustack-worker-nvidia \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].imageID}{"\n"}{end}'
```

If the `imageID` digest is unchanged after a push + restart, you are on the old image.

### What `make package` does and does not touch

- **It does not modify your working tree.** The Dockerfile bind-mounts the *build context* into
  the builder and runs `make build` there, so the version-stamping described below happens on a
  copy.
- **It does pick up uncommitted changes**, because the build context is your working tree (minus
  `.dockerignore`).
- **A dirty tree stamps the image version as `v0.0.0`.** `hack/lib/version.sh` sets
  `GIT_VERSION=v0.0.0` whenever `git status --porcelain` is non-empty, and also whenever the branch
  name is not a valid `vX.Y.Z` (so `feat/*` lands on `v0.0.0` even when clean). Override with
  `VERSION=... make package` if you need a distinguishable version.

!!! danger "`make build` on the host rewrites two tracked files"
    `hack/build.sh` stamps the version into `gpustack/__init__.py` and `pyproject.toml`, then runs
    `git checkout -- <those two files>` to restore them. **Any uncommitted edits you have in
    `gpustack/__init__.py` or `pyproject.toml` are discarded.** And because the script runs under
    `set -e`, a failed build skips the restore entirely and leaves the stamped values behind.
    Commit or back up those two files before running `make build` directly.
    `make package` is unaffected — it only builds inside the container.

## 2. Worker — option B: `hostPath` overlay (no image build)

For fast iteration, mount both working trees into the pod and put them ahead of the installed
packages on `sys.path`. One patch covers `gpustack` **and** `gpustack-runtime`; no image build, no
registry round trip.

```bash
kubectl -n gpustack-system patch ds/gpustack-worker-nvidia --type=strategic -p '
spec:
  template:
    spec:
      volumes:
        - name: src-gpustack
          hostPath:
            path: /path/to/gpustack
            type: Directory
        - name: src-runtime
          hostPath:
            path: /path/to/runtime
            type: Directory
      containers:
        - name: gpustack-worker
          volumeMounts:
            - name: src-gpustack
              mountPath: /src/gpustack
            - name: src-runtime
              mountPath: /src/runtime
          env:
            - name: PYTHONPATH
              value: /src/gpustack:/src/runtime
            - name: PYTHONDONTWRITEBYTECODE
              value: "1"
'
```

`PYTHONDONTWRITEBYTECODE=1` matters: the container runs as root, and without it the interpreter
writes `__pycache__` directories straight into your host working tree, owned by root.

On a single-node cluster the `hostPath` is simply your machine. On a multi-node cluster the
directory must exist on **every** node the DaemonSet lands on — pin it with a `nodeSelector` first.

### Why `PYTHONPATH` reliably wins

CPython builds `sys.path` as: the script/`-m` entry, then the `PYTHONPATH` entries, then the
standard library, then `site-packages` / `dist-packages`. In the worker image the `PYTHONPATH`
entries land at `sys.path[1]` — ahead of both the standard library and the
`/usr/local/lib/python3.11/dist-packages/` copy that `uv pip install` put there at build time.

The one way `sys.path` order could be bypassed is a `sys.meta_path` finder, since `meta_path` is
consulted before `PathFinder` ever looks at `sys.path` — this is how PEP 660 editable installs can
win. The worker image has no such finder: `sys.meta_path` holds only `_distutils_hack` (which
intercepts `distutils`/`setuptools` and nothing else) plus the three built-in importers. So the
overlay is not shadowed.

Confirm both facts inside the pod:

```bash
kubectl -n gpustack-system exec ds/gpustack-worker-nvidia -- \
  python3 -c "import sys; print(sys.path[:4]); print(sys.meta_path)"
```

### Prerequisite: no top-level name may collide with the standard library

Because `PYTHONPATH` sits *before* the standard library, any top-level importable name at a repo
root shadows a stdlib module of the same name — across the whole process, including third-party
code. A repo root that grows a `types.py`, a `queue/` package or similar breaks the container in
ways that look nothing like an import problem.

At the time of writing, the importable top-level names are `conftest`, `gpustack`, `tests`
(gpustack) and `gpustack_runtime` (runtime) — none collide. **Re-run this check whenever a new
top-level module or package appears at either repo root:**

```bash
python3 - <<'EOF'
import os, sys
for root in ("/path/to/gpustack", "/path/to/runtime"):
    names = {
        e for e in os.listdir(root)
        if (os.path.isdir(os.path.join(root, e))
            and os.path.exists(os.path.join(root, e, "__init__.py")))
        or (e.endswith(".py"))
    }
    names = {e[:-3] if e.endswith(".py") else e for e in names}
    print(root, "->", sorted(names & set(sys.stdlib_module_names)) or "no collisions")
EOF
```

### Prerequisite: `gpustack/third_party/bin` must be populated

The overlay redirects package-relative data lookups too. `gpustack/worker/tools_manager.py` resolves
its bundled tools through `pkg_resources.path("gpustack.third_party", "bin")`, so with the overlay
active that resolves inside your working tree — not the image, where the Dockerfile's
`gpustack download-tools` put them. `.dockerignore` excludes `**/third_party/bin`, so a fresh clone
does not have it.

```bash
ls /path/to/gpustack/gpustack/third_party/bin   # expect: fastfetch gguf-parser llama-box versions.json
```

If it is missing, run `make build` (mind the warning above) or `gpustack download-tools` locally
first. The tools must match the container's platform, which on a single-node `linux/amd64` setup
they do.

### Verify the overlay is live

```bash
kubectl -n gpustack-system exec ds/gpustack-worker-nvidia -- \
  python3 -c "import gpustack, gpustack_runtime; print(gpustack.__file__); print(gpustack_runtime.__file__)"
```

Both paths must start with `/src/`. Note this checks a *new* process; to confirm the long-running
worker picked it up, restart the DaemonSet after patching and read the version line in its log
(see below).

### Reverting

```bash
kubectl -n gpustack-system patch ds/gpustack-worker-nvidia --type=json -p '[
  {"op": "remove", "path": "/spec/template/spec/volumes/0"},
  {"op": "remove", "path": "/spec/template/spec/volumes/0"}
]'
```

Index-based JSON-patch removal is brittle — check the current list first with
`kubectl -n gpustack-system get ds/gpustack-worker-nvidia -o jsonpath='{.spec.template.spec.volumes[*].name}'`,
and remove the matching `volumeMounts` and `env` entries the same way. When in doubt, re-download
and re-apply the cluster's `manifest.yaml`, then re-apply your `imagePullPolicy` patch.

### Reading the version to tell which code you are on

`gpustack/__init__.py` carries `__version__` / `__git_commit__`, and `hack/build.sh` rewrites them
at build time. That gives a reliable fingerprint in the worker's `Version check passed: worker ...`
log line:

| Reported version | What you are running |
| --- | --- |
| `0.0.0` (commit `HEAD`) | An un-stamped working tree — the host server, or a `PYTHONPATH` overlay |
| `v0.0.0` | A `make package` image built from a dirty tree or a non-tag branch |
| `vX.Y.Z` | An image built from a clean, tagged tree |

Related detail: `importlib.metadata.version("gpustack")` reads the `.dist-info` that `uv pip install`
wrote into the image, which the overlay does not replace — so it keeps reporting the image's
version even while the *code* comes from `/src`. The two disagreeing is expected under the overlay.

The version mismatch check (`gpustack/utils/version.py`) short-circuits to "compatible" when either
side is exactly `0.0.0`, so a working-tree server and an overlaid worker will not produce a spurious
mismatch warning.

## 3. Operator — build an image or overlay the binary

The operator is Go. Build it first:

```bash
cd /path/to/gpustack-operator
make build          # -> .dist/build/gpustack-operator, for the host OS/arch
```

`hack/build.sh` builds for `${BUILD_OS}/${BUILD_ARCH}` by default; set `BUILD_PLATFORMS=linux/amd64`
if you are cross-building, in which case the output is suffixed
(`.dist/build/gpustack-operator-linux-amd64`).

### Option A: image

```bash
make package                                        # -> gpustack/gpustack-operator:dev
docker tag gpustack/gpustack-operator:dev <your-registry>/gpustack-operator:<tag>
docker push <your-registry>/gpustack-operator:<tag>
kubectl -n gpustack-system set image deploy/gpustack-operator-worker \
  main=<your-registry>/gpustack-operator:<tag>
```

The same `imagePullPolicy` trap applies — `gpustack/k8s/operator.jinja` also hardcodes
`imagePullPolicy: IfNotPresent`. Either use a fresh tag or patch the Deployment to `Always`.

### Option B: `hostPath` over the binary

The image's `ENTRYPOINT` is `tini --` and the args resolve `gpustack-operator` from `PATH`, so
mounting a file over `/usr/bin/gpustack-operator` is enough:

```bash
kubectl -n gpustack-system patch deploy/gpustack-operator-worker --type=strategic -p '
spec:
  template:
    spec:
      volumes:
        - name: operator-bin
          hostPath:
            path: /path/to/gpustack-operator/.dist/build/gpustack-operator
            type: File
      containers:
        - name: main
          volumeMounts:
            - name: operator-bin
              mountPath: /usr/bin/gpustack-operator
'
```

The binary must be `linux/<node-arch>` and executable (`chmod +x`).

!!! note "The binary swap does not swap the bundled chart"
    The operator image also carries a packaged Helm chart at
    `/etc/gpustack/charts/gpustack-operator-<version>.tgz`, built from
    `deploy/gpustack-operator/chart` at image build time. If your change is in the chart rather than
    in Go code, a binary-only overlay will not pick it up — build the image instead.

## Anti-pattern: mutating a running pod

Do **not** verify a change by doing this:

```bash
# WRONG on a DaemonSet / Deployment
kubectl exec -it <pod> -- pip install -e /src/gpustack
kubectl cp ./gpustack/some_module.py <pod>:/usr/local/lib/python3.11/dist-packages/gpustack/
```

A pod's writable layer is not durable. Node pressure, a rollout, a probe failure, a kubelet
restart, or the DaemonSet controller recreating the pod for any reason all discard it, and the
replacement quietly starts the image's original code. There is no event that tells you your
verification target changed underneath you.

Both alternatives above (image, or `hostPath` overlay) live in the pod **spec**, so a recreated pod
comes back with the change intact.

## Shell gotcha: this repo's scripts assume `bash`

`zsh` does not word-split unquoted parameter expansions:

```bash
L="a b c"; for f in $L; do echo "[$f]"; done
# bash -> [a] [b] [c]
# zsh  -> [a b c]      (one iteration)
```

A loop copied from a `bash` snippet runs exactly once in `zsh`, over the whole string, and usually
fails in a way that reads like a bad value rather than a bad split. If you are on `zsh`, either run
such helpers with `bash -c '...'`, use an array (`L=(a b c)`), or add `setopt shwordsplit`.

## Checklist before trusting an end-to-end run

1. `imageID` digest of the worker pod changed since the previous run (image path), **or**
   `gpustack.__file__` reports `/src/...` (overlay path).
2. Worker log's version line matches the path you intended (`0.0.0` for overlay, `v0.0.0` /
   `vX.Y.Z` for an image).
3. No stdlib name collision at either repo root, if you added a top-level module.
4. `gpustack/third_party/bin` populated, if using the overlay.
5. Server's `gpustack_runtime.__file__` points where you expect, if you care about runtime changes.
