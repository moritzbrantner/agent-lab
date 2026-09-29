# Routing evidence v1

Twenty steady-state samples and four separate cold runs compare the same batch
workload on pinned local models and identical hardware/software.

| Policy | Success | Mean generated tokens | Mean primitive model calls |
| --- | --- | --- | --- |
| single cheap | 0/5 | 33 | 1 |
| single strong | 5/5 | 125 | 2 |
| cheap-first | 5/5 | 158 | 3 |
| role-specialized | 5/5 | 381 | 3 |

Routing recovers correctness relative to the cheap baseline but spends more than
using the strong model directly. The role policy pays for a rejected 256-token
cheap final reply. These points do not improve the generated-token Pareto view.
All rejected responses, variance, latency, sampled memory, decisions and child
attempt identities are retained. The policies remain optional experiments.
