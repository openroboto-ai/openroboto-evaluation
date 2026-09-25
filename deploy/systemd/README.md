# Service templates

These files are examples, with placeholders for checkout paths and credentials.
Host-specific production units and populated environment files are not distributed.

- `openroboto-axis-rotation@.service` with `axis-rotation.env.example` supports
  separate instances such as `@dev` and `@production`.
- `openroboto-gpu-monitor.service.example`, `gpu-monitor.env.example` and
  `gpu-monitor.yaml.example` describe optional GPU health notifications.

Copy examples to the locations named in each unit, replace placeholders, and set
credential files to mode `0600`. Keep backend-specific state, model caches, output
and version directories separate. No real credentials should be committed.

Use the [worker guide](../../benchmark_worker/README.md) for `axis_v1.0` queue routing.
Review commands and GPU allocation before enabling an instance.
