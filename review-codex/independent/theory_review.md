# Mathematical review of setup-paper/sections/{2-setup,3-results}.tex

Source: main 982822f554e547cc5c4468b70d5aaaae88356814. The theorem is valid with the
stated independent conditional sampling and action-compatible coarse observation map.
No mathematical counterexample to the theorem under those assumptions was found.
The following supplies the missing proof and makes the scope explicit.

## A precise finite-space statement

Let P be a finite-horizon process on outputs O^n. Given the entire previous output
history z, its next-round law is the product of K_i(.|h_i(z)). Let c:O→A retain an
action, and require that each coarse local state s_i(z) be computable from c(z) alone.
In the paper this follows from q∘rho_i = rho_bar_i∘c^n, including START and the
initial shorter histories. For any fixed stochastic tables kappa_i(.|s), defined
on every reachable coarse state, let Q be the autonomous product-action process
using these tables. They need not be population conditional means for this bound.

Then

TV(P_actions, Q_actions) ≤ min(1, Σ_t Σ_i E_P TV(c_*K_i(.|h_i^t),kappa_i(.|s_i^t))).

For every deterministic measurable trajectory statistic f, the same RHS bounds
TV(f_*P_actions,f_*Q_actions). This is a distributional assertion, not a claim that
any observed error in a statistic is bounded without sampling error. In particular,
binned per-round TV is a coarsening of a marginal; it cannot identify full path TV.

## Well-defined hybrids even when messages have been dropped

A hybrid that merely replaces an action transition but omits signals is not yet a
process on the original history space. Lift each kappa to a kernel on O as follows.
Disintegrate K_i into its action marginal K_i^A and conditional residual output
C_i(o|a,h), supported on outputs with c(o)=a. If K_i^A(a|h)=0, choose any such
conditional distribution. In this project c is onto and the finite fiber exists.
Define K_tilde_i(o|h)=kappa_i(a|s_i(h)) C_i(o|a,h), a=c(o).
The zero-mass choices make the kernel total; they require no division by zero and
do not change the action marginal. Directly summing absolute differences over each
fiber gives TV(K_i,K_tilde_i)=TV(K_i^A,kappa_i).

Use chronological hybrids: the first j rounds use the true kernels, all remaining
rounds use the lifted product kernels. Consecutive hybrids share a true prefix up
to the changed round and the same lifted continuation afterward. Data processing
of the common continuation and product subadditivity give a difference at most
Σ_i E_P TV(K_i,K_tilde_i) at that round. Crucially the prefix has the true P law;
this explains why the expectation is under p_swarm, not an unknown hybrid law.
Sum over rounds, and then project out the signals. In the all-lifted process,
actions conditional on full history have exactly the product kappa law, depending
only on past actions; hence their marginal process is precisely Q.

This is a finite-horizon coupling/telescoping argument. It is not evidence for a
novel general perturbation theorem. No ergodicity, optimisation, rationality,
stationarity of swarm histories, or asymptotic convergence is needed. The tables
may also be time-indexed if the indexing and error definition are changed together.

## Exact transfer: correct with full-history support

For finite H_i, zero population mean of the nonnegative local defect implies zero
defect at every h to which p_test assigns positive probability. If every swarm
history law is absolutely continuous with respect to p_test (ordinary support
inclusion in these finite spaces), every term in the bound vanishes. Thus the
action-trajectory laws agree. Full-history support, rather than only coarse-state
coverage, is the relevant assumption. In a general nonfinite space replace
topological support inclusion by absolute continuity; the current finite theorem
does not need that generalisation.

The project's positive Bernoulli/multinomial probabilities and positive mass at
every length give positive probability to all locally possible finite records,
including START. The default full-history domain therefore has adequate support.
The enumerated last-1 signal table has no fitted-history Monte Carlo error: the
rendered prompt is constant within state. Last-1 counts and signal-aware cheap talk
can both transfer exactly. Last-1 actions-only cheap talk need not transfer exactly,
because marginalising messages can discard predictive feedback. The sentence at
the end of 2-setup.tex about exactness only on counts should mention the retained-
signal comparator and apply only to the actions-only audit if that is the intent.

## Concrete counterexamples to stronger interpretations

1. **Drop conditional independence.** Two agents, one round, shared fair bit B,
   outputs (B,B). Both individual marginals equal the independent audit's fair coin,
   so every marginal defect is zero, yet true and product joint laws have TV=1/2.
   Correlation of agents across rounds or through graph wiring is allowed; shared
   randomness not accounted for by the conditional product law is what fails here.
2. **Replace history support by coarse-state coverage.** One agent, two rounds,
   no neighbours. At START output (L,high). The round-2 action is H after own signal
   high and L after own signal low. Let test histories cover START and the round-2
   history (L,low), assigning each positive mass. The actions-only START and L states
   are both covered and test defect is zero; the composed model stays L but the
   swarm switches to H. The actually visited history was absent from p_test.
3. **Replace a population defect by its finite estimate.** If a fixed local error
   is 1 on a history of test probability eta and 0 elsewhere, m independent audit
   samples report zero with probability (1-eta)^m despite true defect eta. A zero
   held-out error estimate is not an exact-transfer certificate.
4. **Infer path agreement from endpoints.** An iid fair-coin process and an iid
   coin with P(H)=0.55 both mix in one step; their one-time TV is 0.05. The TV of
   their path laws tends to 1 as length increases (the empirical average separates
   them). Ergodic endpoint perturbation bounds cannot be used as a uniform bound
   for entire trajectories. Conversely identical endpoint laws can conceal wholly
   different earlier behaviour, as the project's late all-low outcomes illustrate.

## Population theorem, plug-in expectation, finite estimate

Keep three levels explicit: (i) a theorem about fixed true kernels and fixed tables;
(ii) a numerical RHS using population expectations that are generally unavailable;
(iii) a logged-run estimate of those expectations, with sampling uncertainty.
Finite fitted tables can be used in (i), but their actual local defects are those
of the fitted table. Estimating those defects on independent logged runs is still
not a certificate without a valid upper confidence bound. Bootstrap intervals
are descriptive, particularly for degenerate samples.

For example, conditional on a fixed table independent of the R logged runs, set
Z_r=Σ_t,i e_rit ∈ [0,nT]. Hoeffding supplies the valid but often vacuous statement
TV(P_actions,Q_actions) ≤ min(1, mean(Z)+nT sqrt(log(1/alpha)/(2R)))
with probability at least 1-alpha. This illustrates what an actual data-derived
certificate requires; it is not proposed as a useful new headline bound. The
paper correctly labels the existing bootstrap RHS as an estimate. At Llama n=10
the binary estimate is already 11.98 and the joint estimate 43.14, so clipping
gives 1. Comparing empirical binned-TV growth ratios with uncapped bound growth
cannot establish that the trajectory bound is tight in n.

The expected CRPS advantage Σ_k(Q(k)-G(k))²/(5n) when Y~G is the usual proper-score
identity. Its positivity demonstrates a difference between forecast laws under
the chosen reference, not independent evidence that G is the real swarm law.
