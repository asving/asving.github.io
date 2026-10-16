# MONOLITH_REPORT — end-to-end local certificate for `SurfaceBlowupV2.main_field_form`

Written 2026-09-30 by the certificate agent (Claude Fable 5.1). Everything below was verified by building in
`/data/users/asvin/prove2me_workspace/final/monolith`; nothing is taken from the platform's status flags.

## What this is

The Prove2Me platform marks node N660 (`SurfaceBlowupV2.main_field_form`, Theorem 1.1 with the paper's quantifiers)
as Proved because each of the 1001 theorem nodes of the SurfaceBlowupV2 blueprint has an ACCEPTED or SKETCH_ACCEPTED
submission, each verified *in isolation* against sorry-stubs of its children. This directory is one Lean project in
which every theorem module contains its accepted proof (no stubs), the whole library builds, and the axioms of the main
theorem are checked in a single environment. It also certifies, at kernel level, that each of the 1001 proved theorems has
*exactly* the blueprint stub's statement.

## Headline results (all verified by building)

* `lake build Theorems`: **1001 / 1001 modules built from accepted proofs, 0 stubs left, 0 errors, 0 `sorry` warnings.**
  Wall time **408 s** (6 min 48 s) on 16 cores (`taskset -c 0-15`), log `logs/build_theorems.log`.
* `#print axioms` (`Monolith/Main.lean`, log `logs/main_axioms.log`), verbatim:

```
'SurfaceBlowupV2.main_field_form' depends on axioms: [propext, Classical.choice, Quot.sound]
'SurfaceBlowupV2.main' depends on axioms: [propext, Classical.choice, Quot.sound]
'SurfaceBlowupV2.quartic_totally_ramified_at_two' depends on axioms: [propext, Classical.choice, Quot.sound]
'SurfaceBlowupV2.quartic_projective_totally_ramified_at_two' depends on axioms: [propext, Classical.choice, Quot.sound]
'SurfaceBlowupV2.quartic_ramified_quadratic_at_two' depends on axioms: [propext, Classical.choice, Quot.sound]
'SurfaceBlowupV2.quartic_two_inert_quadratics_at_two' depends on axioms: [propext, Classical.choice, Quot.sound]
'SurfaceBlowupV2.cubic_totally_ramified_at_three' depends on axioms: [propext, Classical.choice, Quot.sound]
'SurfaceBlowupV2.quadratic_monic_formula' depends on axioms: [propext, Classical.choice, Quot.sound]
'SurfaceBlowupV2.quadratic_at_two' depends on axioms: [propext, Classical.choice, Quot.sound]
```

* **Statement identity, kernel-checked (`Monolith/StubCertify.lean`, output `stub_certify.tsv`).** The workspace's stub
  environment (the 1001 blueprint stubs `../../Theorems/Thm_SurfaceBlowupV2_*.lean` as compiled there, 5878 modules) was
  loaded as a second in-memory environment; for every name the Lean kernel (synchronous `Kernel.Environment.addDecl`,
  `debug.skipKernelTC = false`) was asked to accept `theorem <name>.stub_certified : <stub's exact elaborated type> :=
  <monolith theorem>`. Result: **KERNEL_OK for 1001 / 1001**. 999 of the 1001 stub types are alpha-identical
  (`Expr ==`) to the monolith theorem's type; the remaining 2 (`main_field_form`, `product_cell_type_event_sets`)
  differ only in an instance-argument path chosen by typeclass synthesis (the solutions import more modules than the stubs:
  `IsDomain K` via `instIsDomain` vs `IsDomain.of_isSimpleRing` — a Prop, equal by proof irrelevance; `PartialOrder (Set _)` via
  `ChainCompletePartialOrder.toPartialOrder` vs `CompletePartialOrder.toPartialOrder` — two projections of the same `Set` lattice) and
  were accepted by kernel definitional unfolding. Level parameters agree for all 1001.
* **Axioms of all 1001 theorems** (same run, and `typedump_monolith.tsv` / `type_compare.json`): every one depends only on
  a subset of {`propext`, `Classical.choice`, `Quot.sound`} — 978 use all three, 14 `[Quot.sound, propext]`, 4 `[propext]`,
  1 `[Classical.choice, propext]`, 2 none. No `sorryAx`, no other axiom, anywhere.
* **Textual header identity (`header_check.py`, `header_check.json`).** Stub header (binders + type, minus `by sorry`) as a
  prefix of the monolith theorem's text: **972 byte-identical, 29 identical up to whitespace,
  0 identical only after deleting all whitespace, 0 mismatches.** All 29 whitespace cases are the same thing:
  a term-mode proof puts a newline after `:=` where the stub has `:= by sorry` on one line.
* **Source provenance.** All 1001 accepted solutions were downloaded from the server (`GET /submissions/<id>/solution`,
  sequential, cached in `src/<node>.lean`, `src/_download_meta.json`); **all 1001 are byte-identical to the local
  `../../Solutions/Sol_SurfaceBlowupV2_<slug>.lean`** (`diff_check.py`). N917's downloaded source imports
  `Theorems.Thm_SurfaceBlowupV2_pattern_divisors_equiv_colorings_v2`; the deprecated `pattern_divisors_equiv_colorings`
  statement does not occur in the monolith (no such module; the string appears only inside a comment of the v2 module).
* **sha256** of the main-theorem module and of its source:

```
11e67ef68fe819bcb97866d01452c550457db096bdf9715566a60410a2f40b91  Theorems/Thm_SurfaceBlowupV2_main_field_form.lean
c7ba2254ab2bfcbaeeeaf8f728927940fb0f0ae2ed4eb3938532cdadbf895720  src/N660.lean
c7ba2254ab2bfcbaeeeaf8f728927940fb0f0ae2ed4eb3938532cdadbf895720  ../../Solutions/Sol_SurfaceBlowupV2_main_field_form.lean
```
  `SHA256SUMS` lists all 1001 transformed modules (sha256 of that file: `12eabb3a8d0d1be2140490bd0c0571ecf080a6fba8af6bc2e54723a36e3fedf3`).
* Post-edit re-check of the project (`logs/rebuild_check.log`): `REBUILD_EXIT=0`.

## How the modules were produced (mechanical; proof text untouched)

`transform.py` maps each accepted `theorem solution` file to `Theorems/Thm_SurfaceBlowupV2_<slug>.lean`:
imports, `set_option` and `open` preamble kept verbatim; `theorem solution` renamed to `theorem SurfaceBlowupV2.<slug>`;
proof bodies never edited. Helper declarations preceding the main theorem (present in 158 of 1001 files; the other
843 files are the renamed theorem alone) are wrapped in `namespace SolN<id> … end SolN<id>` with `open SolN<id> in`
placed before the main theorem's command (i.e. before any docstring / `set_option … in` / `open … in` attached to it).
Special handling, all recorded in `transform_manifest.json`: depth-0 `open` commands inside the helper region are re-emitted
after the namespace (N947, N1108); N1108's file-level `noncomputable section` is hoisted so the namespace nests inside it;
files declaring helpers under `SurfaceBlowupV2.` get an extra `open SolN<id>.SurfaceBlowupV2 in` (N092, N725, N739, N982, N1132).
No module needed a proof change; **no stub was left in place; defects: none.**

Everything the transform does is visible as a diff between `src/<id>.lean` and the corresponding `Theorems/` module.

## Project layout and reproduction

* `lakefile.lean`, `lean-toolchain` (v4.33.1), `lake-manifest.json`: copied from the workspace (same Mathlib pin
  `0df444a…`). `.lake/packages/` is a copy of the workspace's built dependency tree; `Definitions/` is a verbatim copy of the
  workspace's 63 definition modules, rebuilt here from source (`logs/build_defs.log`).
* `Theorems/` — the 1001 transformed modules. `src/` — the downloaded accepted sources. `Monolith/Main.lean` — the
  `#print axioms` file. `Monolith/TypeDump.lean` — dumps kind / level params / hash / axioms / `repr` type of all 1001 names
  (`TYPEDUMP_OUT=<file> lake env lean Monolith/TypeDump.lean`; run here → `typedump_monolith.tsv`, and in the workspace →
  `typedump_workspace.tsv`; `compare_types.py` diffs them → `type_compare.json`). `Monolith/StubCertify.lean` — the kernel
  certifier (`WS_LEAN_PATH="$(cd ../.. && lake env printenv LEAN_PATH)" CERTIFY_OUT=<file> lake env lean Monolith/StubCertify.lean`).
* Reproduce: `taskset -c 0-15 lake build Theorems && lake env lean Monolith/Main.lean`.
* Machine etiquette: all Lean processes were affinity-bound to cores 0–15; Lake's default job count is not bounded by
  `taskset`, so up to ~95 `lean` processes shared those 16 cores (load average rose accordingly; no other cores were used).

## Trust boundary

Lean checks proofs, not statements. This certificate shows the 1001 accepted proofs compose into one sorry-free build
and that each proves *exactly* the blueprint stub's elaborated statement. Whether the stubs and the 63 definition modules
say what the paper means remains the human-review item listed in `blueprint/COMPLETE.md`.
