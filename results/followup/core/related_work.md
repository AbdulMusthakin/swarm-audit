# Related work: prior results, our additional test, remaining overlap

**Status of the sources (8 Nov 2026, about 01:00 UTC).** arxiv.org is not reachable from this
session's network, so no full text was read. Titles, authors and dates below come from web
search results that index each arXiv page. The summary of each paper's findings comes from
its indexed abstract or snippets only. **Methods were not read, so no claim of a
methodological distinction should be made from this table** until an author has read the
papers. No "first" claim is supported by this check.

| Paper (verified identity) | What it reports (from abstract / snippets only) | What our experiment adds | Overlap to check before claiming a difference |
|---|---|---|---|
| A. Flint, L. M. Aiello, R. Pastor-Satorras, A. Baronchelli, *Group size effects and collective misalignment in LLM multi-agent systems*, arXiv:2510.22422 (submitted 25 Oct 2025, cs.MA). A search result lists PNAS 123(34) e2531697123 (18 Aug 2026); **venue not verified**. | Interaction can strengthen or create collective biases and override individual preferences. Group size has non-linear, model-dependent effects. A **mean-field analysis** matches simulations above a critical population size, giving deterministic predictions and basins of attraction. | An audit built only from isolated per-agent tests (a tabulated kernel on synthetic histories), composed into an autonomous forecast. It is scored against live swarms with a paired CRPS contrast between two information sets (actions-only and signal-aware), with calibration gates and a decomposition of where the forecast loses information (state coarsening vs history length). | **High.** Their mean-field prediction is also a forecast of collective dynamics from individual-level behaviour. Read how they build the individual policy (isolated queries? measured inside the group?), and whether they report transient (pre-equilibrium) errors. Our clock-indexed audit addresses history length, which a mean-field over stationary policies would also miss. |
| *Cheap Talk Stabilizes Strategic Interaction in LLM Agents*, arXiv:2609.16270. **Authors not confirmed** by search (the hand-off names Lorè, Zhu and Heydari). A related earlier paper is N. Lorè and B. Heydari, *Communication Enhances LLMs' Stability in Strategic Thinking*, arXiv:2602.06081 (4 Feb 2026). | Four open-weight 7-9B models, repeated play in Prisoner's Dilemma, Snowdrift, Stag Hunt and Harmony. Cheap talk stabilises play in 71 of 96 model-game-context comparisons after correction, and 5 reverse under social or team framings. The 2602.06081 paper reports that pre-play messages smooth cooperation trajectories in 10-round PD, with model, prompt and framing dependence. | We do not test whether messages stabilise play. We test whether an isolated audit that conditions on messages forecasts the group better than one that ignores them. It does not at the endpoint (primary null). In early rounds it forecasts slightly worse than the actions-only audit. Our messages are statements about the agent's next bid on a network. | **Medium.** Both use cheap talk among small open-weight models in repeated games. Check whether they model message content as a predictor of later actions; that would overlap with our signal-aware table. |
| H. Madmoun and S. Lahlou (MBZUAI), *Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches*, arXiv:2510.05748 (v1 7 Oct 2025; revised Jan 2026, v3 per the hand-off; listed as EACL 2026 short paper 23). | In a 4-player Stag Hunt, a one-word cheap-talk channel raises coordination from 0% to 48.3%. A curriculum through progressively complex games lowers payoffs by 27.4% in an iterated public-goods game with punishment ("learned pessimism"). | Same as above: forecast adequacy of isolated audits, not whether communication raises cooperation. | **Low to medium.** It is a precedent that one-word cheap talk changes LLM group outcomes. Cite it for that effect, not for auditing. |

**Suggested contribution wording (for the authors to check against the papers):** "We ask
when an audit built from isolated per-agent tests forecasts a networked LLM swarm, what
information it loses, and how to diagnose that loss." Do not write "messages improve
cooperation" or "messages stabilise LLM play" as findings of ours; both have close
precedents above.

Sources (search results used):

- https://arxiv.org/abs/2510.22422
- https://arxiv.org/html/2510.22422v1
- https://comdig.cssociety.org/2026/08/28/group-size-effects-and-collective-misalignment-in-llm-multi-agent-systems/
- https://arxiv.org/pdf/2609.16270
- https://arxiv.org/abs/2602.06081
- https://arxiv.org/abs/2510.05748v1
- https://arxiv.org/html/2510.05748v2
- https://preview.aclanthology.org/eacl-dois/2026.eacl-short.23
