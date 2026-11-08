# Diagnostic sensitivity within the four-agent OLMo setting: results

Exploratory and model-based (plan: `plan.md`, committed before running).

Cutoffs: S1a 0.1906 (original); S1b 0.1918; S2 min tail p <= 0.005200; S3 at least 1 persistent trajectory of 50.

| test | false-positive rate (null) | rejection rate (alternative) |
|---|---|---|
| S1a_shared_max_TV_original_cutoff | 0.0547 [0.0516, 0.0579] | 0.3267 [0.3202, 0.3332] |
| S1b_shared_max_TV_recalibrated | 0.0534 [0.0504, 0.0567] | 0.3260 [0.3195, 0.3325] |
| S2_per_round_tail_joint | 0.0534 [0.0504, 0.0567] | 0.9970 [0.9961, 0.9977] |
| S3_persistence_event | 0.0039 [0.0031, 0.0049] | 0.9997 [0.9993, 0.9999] |

Live cohorts (illustration only):

- original: max TV 0.160 (round 20), min tail p 0.000050 (first reached in round 7; S2 flags rounds 5-20), persistent runs 8; S1a_shared_max_TV_original_cutoff: not flagged, S1b_shared_max_TV_recalibrated: not flagged, S2_per_round_tail_joint: flagged, S3_persistence_event: flagged.
- fresh: max TV 0.180 (round 20), min tail p 0.000050 (first reached in round 6; S2 flags rounds 5-20), persistent runs 9; S1a_shared_max_TV_original_cutoff: not flagged, S1b_shared_max_TV_recalibrated: not flagged, S2_per_round_tail_joint: flagged, S3_persistence_event: flagged.
