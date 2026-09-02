# AMENDMENT v3.2

- parent_protocol_hash: `8014a94241779ca55745ebcf533784a51682a6ff8cfa1ad41af0ce84760e61ce`
- protocol_hash: `0e138a8c817ac355169102decf6ae891ffd2488a0bca82b320e8baf8c1a0f0ad`
- v3.1.17 remains immutable and all 35 old checkpoints remain archived.
- `ppo_mlp` is excluded from v3.2 evaluation and replaced by `traditional_ppo`.
- Only the new traditional PPO is trained; six parent variants are reused.
- Formal evaluation is frozen at 21,648 rows with all four ablations on all 144 real tasks.
- Robustness separates known domain shifts from hidden model/perception mismatch.
- Maps are the primary independent statistical units.
