# System comparison (n=987 paired items, reference = blackboard)

| system | accuracy | hallucination | abstention | halluc. among answered | LLM calls / question | Δ halluc. vs ref [95% CI] | Δ acc. vs ref [95% CI] | fixed / broken vs ref (McNemar p) |
|---|---|---|---|---|---|---|---|---|
| no mitigation (raw) | 13.0 | 25.8 | 61.2 | 66.6 | 0 | | | |
| blackboard | 60.0 | 14.6 | 25.4 | 19.6 | 3.45 | reference | |  |
| pipeline | 55.2 | 33.5 | 11.2 | 37.8 | 1.44 | +18.9 [+16.2, +21.7] | -4.8 [-7.7, -1.8] | 19 / 206 (p=0.0000) |
| cove | 20.8 | 38.3 | 40.9 | 64.8 | 4.46 | +23.7 [+20.3, +27.3] | -39.2 [-42.7, -35.7] | 64 / 298 (p=0.0000) |
| cove (abstentions kept) | 14.0 | 13.8 | 72.2 | 49.6 | 1.40 | -0.8 [-3.4, +1.9] | -46.0 [-49.3, -42.8] | 91 / 83 (p=0.5958) |
| pipeline (abstentions kept) | 13.8 | 16.6 | 69.6 | 54.7 | 0.79 | +2.0 [-0.7, +4.6] | -46.2 [-49.5, -42.9] | 77 / 97 (p=0.1495) |

Δ = system minus reference; negative Δ hallucination = fewer hallucinations than the reference. Fixed/broken count items the reference got wrong and the system did not, and vice versa.
