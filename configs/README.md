# Campaign profiles

Every file here is an **example profile**. None of them contains a credential, and
none of them is a measurement.

| File | `mode` | `materialized` | Purpose |
| --- | --- | --- | --- |
| `offline-demo.json` | `offline` | `true` | Fixture-only campaign, intended for the routine CI workflow once fixture transcripts land. Safe to run at any time. |
| `pilot.json` | `live_authorized` | `false` | Declared 600-item direct pilot + 50-task agent pilot. Requires an operator spending cap. |
| `full.json` | `live_authorized` | `false` | Full declared official splits. Requires the whole split to be enumerated. |

No fixture transcripts exist yet, so `offline-demo.json` validates but does not run
yet; that lands with the transport adapter in G03 (T03A). The pilot and full profiles
are inert for a different reason: their item selection has not run.

## Rules these profiles encode

- **`mode`** decides whether a provider may be contacted at all. `offline` makes
  network dispatch structurally impossible; `live_authorized` additionally requires
  `authorization.spending_cap_usd` to be a non-null number.
- **`materialized: false`** means the frozen item selection has not run yet. Such a
  profile cannot be dispatched even in a live campaign: there are no item IDs. The
  `declared_item_count` / `expected_item_count` fields are *targets recorded in
  advance*, never results.
- **`item_ids: []`** is deliberately empty rather than filled with guesses. G05 (T05A)
  performs the deterministic selection against a verified dataset revision; the
  resulting IDs change the manifest hash and are what any run is bound to.
- **No credentials belong here.** `credential_ref` is a name that resolves to an
  environment variable at run time; a secret value serialized into a manifest is a
  schema error (G01) and a redaction failure (G02).
- **Unknown price is `null`, not `0`.** `limits.require_cost_bounds` is true in every
  profile, so an endpoint with no reliable cost bound refuses spending-capped
  execution instead of being treated as free.

## Identity labels are separate

Each endpoint carries `label_provider`, `label_family`, `label_exact_version` and
`label_tokenizer` independently. The first three are *unknown* for an anonymous
alias and stay `null`. A shared tokenizer is tokenization evidence only — different
models ship the same tokenizer — so it is never sufficient for an identity claim.

## Adding a profile

Copy `offline-demo.json` for fixtures, or `pilot.json` for a new live campaign. Give
the campaign a new `campaign_id`; campaign identity is part of the manifest hash, and
a new campaign is how a rerun stays distinguishable from the report it preserves.
