# Contributing

```bash
make install
make lint test
```

- Formatting and lint are `ruff`; `pre-commit` runs both.
- Keep comments for the non-obvious. Names should carry the rest.
- Numerics changes need `make parity` to stay green, or an entry in `docs/legacy-map.md` saying why not.
- Pipeline flags belong in `configs/profiles/*.toml`, not in code defaults.
- New stages return `id,tvt` frames and write their reports next to `submission.csv`.
