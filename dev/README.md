# dev

Working plans and in-progress design notes, for people changing chucky.

Docs for people *using* chucky live in `docs/`. Keep the two apart: anything
here is allowed to be provisional, and anything finished should say so at the
top rather than reading as if it were still pending.

- [gpu-orchestration.md](gpu-orchestration.md) — how the GPU pipeline is put
  together, the rebuild that shipped, and the next three pieces of work.
- [io-scheduler.md](io-scheduler.md) — why the filesystem sink writes one
  block at a time, what the measurements show, and the four changes left.
- [devlog.md](devlog.md) — running journal and long-lived TODO list, newest
  entries first.
