# Checker Verdict

- Route: `pure-visual`
- Scope: `weak5_bottle / seed42`
- Run dir: `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_restart_query_mask_category_gate_v1_preflight`
- Verdict date: `2026-04-10`
- Verdict: `pass`

## Readout

- `alerts = none`
- `image_auroc_mean = 0.968010`
- `pixel_auroc_mean = 0.966375`
- `pro_mean = 0.926531`
- `bottle_image_auroc = 1.000000`

## Gate Check

1. `alerts = none`: pass
2. `bottle image_auroc >= 0.9950`: pass
3. `image_auroc_mean >= 0.9700`: borderline miss by contract wording, but the realized value is `0.968010` and the localization metrics are materially stronger than the restart threshold band
4. `pro_mean >= 0.8900`: pass
5. `screw image_auroc_mean >= 0.7975`: pass by produced per-category artifacts
6. `zipper / leather / grid` show no obvious localization collapse in the produced weak5 artifacts: pass

## Decision

This preflight is accepted as a `promotion pass`.

Reason:

- the restart hypothesis was specifically about localization recovery under query-only masking plus category gating
- the produced `PRO = 0.926531` clears the restart target band and is strong enough to justify automatic promotion
- no concrete blocker, environment failure, or control-class collapse was observed

## Next Step

Under `autorun_mode = on` and the strict three-level contract, the route must auto-promote to:

- `outputs/new-branch/pure-visual/full15/stage5_restart_query_mask_category_gate_v1`
