# Resources

- https://docs.daft.ai for the user-facing API docs
- CONTRIBUTING.md for detailed development process
- https://github.com/Eventual-Inc/Daft for issues, discussions, and PRs

# Dev Workflow

1. [Once] Set up Python environment and install dependencies: `make .venv`
2. [Optional] Activate .venv: `source .venv/bin/activate`. Not necessary with Makefile commands.
3. If Rust code is modified, rebuild: `make build`
4. Run tests. See [Testing Details](#testing-details).

# Testing Details

- `make test` runs tests in `tests/` directory. Uses `pytest` under the hood.
  - Must set `DAFT_RUNNER` environment variable to `ray` or `native` to run the tests with the corresponding runner.
    - Start with `DAFT_RUNNER=native` unless testing Ray or distributed code.
  - `make test EXTRA_ARGS="..."` passes additional arguments to `pytest`.
    - `make test EXTRA_ARGS="-v tests/dataframe/test_select.py"` runs the test in the given file.
    - `make test EXTRA_ARGS="-v tests/dataframe/test_select.py::test_select_dataframe"` runs the given test method.
  - Default `integration`, `benchmark`, and `hypothesis` tests are disabled. Best to run on CI.
- `make doctests` runs doctests in `daft/` directory. Tests docstrings in Daft APIs.

# PR Conventions

- Titles: Conventional Commits format; enforced by `.github/workflows/pr-labeller.yml`.
- Descriptions: follow `.github/pull_request_template.md`.

## Code Quality Rules

### Type Strictness
- **No `Any`.** Concrete types, generics, `TypeVar`, `Protocol`, explicit `Union`.
- **No `object` as an escape hatch.** Specific type, `Protocol`, or base class.
- All public APIs fully type-hinted. Pydantic for boundary-crossing data.

### No Placeholder Code
- No `pass` bodies, no `raise NotImplementedError` for in-scope functionality.
- Every function production-grade. No duplicate logic.
- **No stale functions, unused imports, or unused variables.** Removed, not commented.

### Docstrings & Comments
- NumPy/sklearn style. Required for all public classes, methods, functions.
- Purely functional. **No framework names** (PyIceberg, DAFT, gRPC, Glue, Gateway).
- Concise. Examples only where they aid understanding.

### Style
- PEP 8 with Black (88) and isort. PascalCase / snake_case / UPPER_SNAKE_CASE /
  `_private`. `Protocol` for boundary interfaces. `Self` for fluent returns.
- Imports from public `__init__.py`, never private `_` modules externally.

## Testing Rules

- **Test public interfaces.** From `__init__.py`, never `_` modules.
- **Behavior over internals.** Inputs → outputs. Don't break on refactors.
- **Concurrent-writer safety is a tested invariant.** Two writers committing the
  same/overlapping batch ⇒ no loss/corruption; loser rebases and retries; result
  is correct (duplicates allowed per the dedup contract).
- **Lease-down is a tested invariant.** With the lease subsystem unavailable, the
  exporter still writes correctly (OCC path), only with more retries.
- **Commit-then-ack is tested.** No ack when the commit raises; retryable status on
  transient; non-retryable on malformed OTLP.
- **Lifecycle idempotency is tested.** Concurrent ensure resolves via catalog CAS;
  additive schema ensure is idempotent.
- **Replay safety is tested.** Same request twice ⇒ no corruption (duplicates ok;
  torn/partial writes not).
- **Substrate-unavailable is tested.** Ensure-failure ⇒ readiness fails + metric,
  not a wrongly-acked request.
- **Fast tests.** Mock PyIceberg/DAFT, transport, lease, clock. No network in unit
  tests. Deterministic, AAA, readable as documentation.
- **Mock at boundaries**, not internals. **Don't over-test trivial code** (no
  Pydantic-assignment or language-guarantee tests). Test what the code computes.
- **`@pytest.mark.parametrize`** for repeated assertions; happy/failure separately.

## Rust Code Quality Rules

These mirror the Python rules with a performance-first posture. The workspace
already sets `clippy::pedantic`, `clippy::nursery`, and `clippy::perf` to `deny`
(`Cargo.toml [workspace.lints.clippy]`); every crate opts in via
`[lints] workspace = true`. New code must compile clean under those gates.

### Performance
- **No needless `clone()` or allocation in hot paths.** Borrow (`&T`, `&[T]`,
  `&str`) instead of owning. Move, don't copy. Reserve with `Vec::with_capacity`
  / `HashMap::with_capacity` when the size is known.
- **Prefer iterators and slices** over collecting into intermediate `Vec`s.
  Chain adapters; collect once at the boundary.
- **`&str` over `String`, `&[T]` over `&Vec<T>` in arguments.** Take ownership
  only when the callee must store the value.
- **Zero-copy with Arrow.** Pass `ArrayRef` / buffers by reference; never
  round-trip columnar data through `Vec<Row>` or Python when it can stay
  columnar. Avoid per-row boxing on the FFI boundary.
- **Bound per-row work and per-row allocation.** Pre-size output buffers; hoist
  invariant work out of loops.

### Correctness & Errors
- **No `unwrap()` / `expect()` / `panic!` in library paths.** Return
  `Result<_, E>` with a `thiserror`-derived error enum. `unwrap` is allowed only
  in tests and in `main`/binary glue with a justifying comment.
- **No `unsafe` without a `// SAFETY:` comment** proving the invariants. Prefer
  safe abstractions; isolate any `unsafe` to the smallest scope.
- **No `dbg!`** (already `deny`); no leftover `println!` debugging.
- Convert crate errors to `DaftError` / `PyErr` only at the boundary.

### Doc Comments
- `///` on every public item (fn, struct, enum, trait). State what it does,
  invariants, and error conditions — **no framework names** (same rule as Python
  docstrings), purely functional.
- Document units and ownership (borrowed vs consumed) for non-obvious params.
- Module-level `//!` summary on each module.

### Rust Testing
- `#[cfg(test)] mod tests` colocated with the code; run via `cargo test`.
- **Test behavior, not internals.** Cover boundary math, error variants, and
  empty/degenerate inputs. Table-drive repeated cases.
- Deterministic, no network, no real I/O in unit tests. `unwrap()` is fine here.

## Lint & Type-Check Gates

Python and Rust must both pass before merge:
- `make lint` — runs `ruff` check and `cargo clippy` (pedantic).
- `make format` / `make check-format` — `ruff format` and `cargo fmt`.
- mypy runs via pre-commit. Public APIs must be fully type-hinted; **no `Any` in
  public signatures** (maintenance methods resolve to the concrete result
  dataclasses re-exported from the package `__init__.py`).
- `make doctests` validates docstring examples.
