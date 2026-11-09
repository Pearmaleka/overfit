# Methods

This note states what each function computes. Notation: $r_{k,t}$ is the return of
strategy $k = 1,\dots,K$ in period $t = 1,\dots,n$, $b_t$ a benchmark return, and
$d_{k,t} = r_{k,t} - b_t$. Sharpe ratios are per period unless annualised by
$\sqrt{\text{periods per year}}$.

## 1. Inference for one Sharpe ratio

`sharpe_ratio_se(returns, method=...)` returns the standard error of
$\widehat{SR} = \hat\mu / \hat\sigma$.

| method | variance of $\widehat{SR}$ | assumption |
|---|---|---|
| `iid` | $(1 + \tfrac12 SR^2) / (n-1)$ | normal, independent (Lo, 2002) |
| `nonnormal` | $(1 - \gamma_3 SR + \tfrac{\gamma_4 - 1}{4} SR^2) / (n-1)$ | independent, skewness $\gamma_3$, kurtosis $\gamma_4$ (Mertens, 2002) |
| `hac` | $\nabla g^\top \hat\Omega \nabla g / n$ | serially dependent |

For `hac`, $SR = g(m_1, m_2) = m_1 / \sqrt{m_2 - m_1^2}$ with $m_j = E[r^j]$, so
$\nabla g = (m_2 / \sigma^3,\ -m_1 / 2\sigma^3)$, and $\hat\Omega$ is the Newey-West
long-run covariance of $(r_t, r_t^2)$ with Bartlett weights and bandwidth
$\lfloor 4 (n/100)^{2/9} \rfloor$. Under iid normality this reduces to Lo's formula.

**Probabilistic Sharpe ratio** (Bailey and Lopez de Prado, 2012): the probability that
the true Sharpe ratio exceeds $SR^*$,

$$
\mathrm{PSR}(SR^*) = \Phi\left( \frac{(\widehat{SR} - SR^*) \sqrt{n-1}}
{\sqrt{1 - \gamma_3 \widehat{SR} + \tfrac{\gamma_4 - 1}{4} \widehat{SR}^2}} \right).
$$

**Minimum track record length**: the $n$ at which $\mathrm{PSR}(SR^*) = 1 - \alpha$,
$\; n^* = 1 + \left(1 - \gamma_3 \widehat{SR} + \tfrac{\gamma_4 - 1}{4} \widehat{SR}^2\right)
\left( z_{1-\alpha} / (\widehat{SR} - SR^*) \right)^2.$

## 2. Selection bias in the best of many backtests

If $N$ strategies without skill are tried and their Sharpe estimates are roughly iid
$N(0, V)$, the expected maximum is

$$
E\left[\max_k \widehat{SR}_k\right] \approx \sqrt{V}\left((1-\gamma)\,\Phi^{-1}\left(1 - \tfrac1N\right)
+ \gamma\, \Phi^{-1}\left(1 - \tfrac{1}{Ne}\right)\right),
$$

with $\gamma \approx 0.5772$ (Euler-Mascheroni). `expected_max_sharpe(method="exact")`
instead integrates $E[M] = \int_0^\infty (1 - \Phi(x)^N)dx - \int_{-\infty}^0 \Phi(x)^N dx$,
which is also defined for a fractional, effective $N$.

**Deflated Sharpe ratio** (Bailey and Lopez de Prado, 2014):
$\mathrm{DSR} = \mathrm{PSR}(SR_0)$ with $SR_0$ the expected maximum above and $V$ the
cross-sectional variance of the trials' Sharpe ratios. The test reproduces the paper's
worked example (DSR = 0.9004).

**Effective number of trials.** Correlated variants are not independent trials.
`effective_number_of_trials` uses the eigenvalues $\lambda_i$ of the correlation matrix
of the trials' returns:
Li and Ji (2005) $\sum_i \mathbb{1}[\lambda_i \ge 1] + (\lambda_i - \lfloor\lambda_i\rfloor)$,
Nyholt (2004) $1 + (N-1)(1 - \mathrm{Var}(\lambda)/N)$ and
Galwey (2009) $(\sum_i \sqrt{\lambda_i})^2 / \sum_i \lambda_i$.

**Haircut Sharpe ratio** (Harvey and Liu, 2015): map $\widehat{SR}$ to
$t = \widehat{SR}\sqrt n$, adjust its one-sided p-value for $N$ tests (Bonferroni or
Sidak), and map the adjusted p-value back to a Sharpe ratio.

**Minimum backtest length** (Bailey, Borwein, Lopez de Prado and Zhu, 2014). Over $y$
years, the annualised Sharpe ratio estimate of a strategy without skill has a standard
deviation of about $1/\sqrt{y}$, so the best of $N$ independent trials is expected to
reach $E[\max_N]/\sqrt{y}$, where $E[\max_N]$ is the expected maximum of $N$ standard
normals. `min_backtest_length` solves for the number of years at which this equals a
target annual Sharpe ratio $\overline{SR}$:

$$
\text{MinBTL} = \left(\frac{E[\max_N]}{\overline{SR}}\right)^2 < \frac{2 \ln N}{\overline{SR}^2}.
$$

With 45 trials, a Sharpe ratio of 1 needs about five years of data before it is more
than the search alone would produce, which is the paper's example. `audit` reports the
MinBTL for the best strategy's Sharpe ratio and the effective number of trials.

## 3. Data-snooping tests

All three tests share one bootstrap. `bootstrap_performance` draws $B$ resamples of the
time index and applies the same draw to every strategy, so the dependence between
strategies is kept. Let $\hat\theta_k$ be strategy $k$'s performance relative to the
benchmark (mean of $d_{k,t}$, or $SR_k - SR_b$) and $\theta^*_{k,b}$ its value in
resample $b$.

**Stationary bootstrap** (Politis and Romano, 1994). Blocks start at uniform positions,
have geometric lengths with mean $L$ and wrap around the sample end. The default $L$ is
the median of the Politis-White (2004) estimates, with the Patton, Politis and White
(2009) correction, over the strategies with the largest means. The implementation
matches `arch.bootstrap.optimal_block_length` exactly.

Bootstrap means are computed without building resampled data sets: with $C$ the
$B \times n$ matrix of how often each observation appears in each resample,
$\bar d^* = C D / n$ is a single matrix product.

**White's Reality Check** (2000). Null: $\max_k E[\theta_k] \le 0$. Statistic
$V = \sqrt n \max_k \hat\theta_k$, null draws $V^*_b = \sqrt n \max_k(\theta^*_{k,b} - \hat\theta_k)$,
p-value $(1 + \#\{V^*_b \ge V\}) / (B + 1)$.

**Hansen's SPA test** (2005). Studentise with
$\hat\omega_k^2 = \hat\gamma_{k,0} + 2\sum_{i=1}^{n-1}\kappa(n,i)\hat\gamma_{k,i}$,
$\kappa(n,i) = (1 - \tfrac in)(1-q)^i + \tfrac in (1-q)^{n-i}$, $q = 1/L$, which is the
exact variance of $\sqrt n\,\bar d^*$ under the stationary bootstrap. The statistic is
$T = \max(0, \max_k \sqrt n \hat\theta_k / \hat\omega_k)$ and the null draws are
$T^*_b = \max(0, \max_k \sqrt n(\theta^*_{k,b} - \mu_k)/\hat\omega_k)$, with three choices
of $\mu_k$:

- upper: $\mu_k = \hat\theta_k$ (every strategy treated as if exactly at the boundary);
- consistent: $\mu_k = \hat\theta_k \mathbb{1}[\hat\theta_k \ge -\hat\omega_k\sqrt{2\log\log n / n}]$;
- lower: $\mu_k = \max(\hat\theta_k, 0)$.

The consistent version stops strategies that are clearly worse than the benchmark from
inflating the null distribution, which is why adding poor strategies does not destroy
its power the way it does for the Reality Check.

**Romano-Wolf step-down** (2005, 2016). Tests every $H_k: \theta_k \le 0$ and controls
the family-wise error rate. With $t_{(1)} \ge \dots \ge t_{(K)}$ the sorted studentised
statistics and $t^*_{k,b} = \sqrt n(\theta^*_{k,b} - \hat\theta_k)/\hat\omega_k$, the
adjusted p-value of the $j$-th strategy is

$$
p^{\text{adj}}_{(j)} = \max_{i \le j}\ \frac{1 + \#\{b : \max_{l \ge i} t^*_{(l),b} \ge t_{(i)}\}}{B + 1}.
$$

`recentre="consistent"` replaces $\hat\theta_k$ by the consistent $\mu_k$ in $t^*$,
which is the step-wise SPA test of Hsu, Hsu and Kuan (2010).

## 4. Probability of backtest overfitting

`probability_of_backtest_overfitting` implements combinatorially symmetric
cross-validation (Bailey, Borwein, Lopez de Prado and Zhu, 2017). Split the $n \times K$
matrix of returns into $S$ contiguous blocks. For each of the $\binom{S}{S/2}$ ways to
pick half of them as the training set, select the strategy with the best training
Sharpe ratio, compute its rank $\omega \in (0, 1)$ among all strategies on the other
half and its logit $\lambda = \log(\omega / (1-\omega))$. PBO is the share of splits
with $\lambda \le 0$: the training winner does no better than the median out of sample.

The metric is computed from per-block sums $\sum_t r$ and $\sum_t r^2$, so each split
costs $O(SK)$ rather than $O(nK)$, and all 12,870 splits for $S = 16$ are a few matrix
products.

## 5. Cross-validation with overlapping labels

A label computed over $[t_0, t_1]$ shares information with every sample whose label
interval overlaps it (Lopez de Prado, 2018, ch. 7). For a test fold spanning
$[t_0^{\text{test}}, t_1^{\text{test}}]$, `PurgedKFold` removes every training sample
with $t_{0,i} \le t_1^{\text{test}}$ and $t_{1,i} \ge t_0^{\text{test}}$ (purging), and
the $\lfloor h n \rfloor$ samples that start right after the fold (embargo $h$).

`CombinatorialPurgedKFold` divides the sample into $N$ groups and tests on each of the
$\binom{N}{k}$ combinations of $k$ groups. Each group is tested
$\binom{N-1}{k-1}$ times; `paths()` assigns those test predictions to
$\binom{N-1}{k-1}$ complete out-of-sample paths, and `backtest_paths()` stitches the
per-split predictions into an array with one row per path:

```python
cv = CombinatorialPurgedKFold(6, 2, t1=label_times(len(X), horizon=20), embargo=0.01)
preds = [model.fit(X[train], y[train]).predict(X[test]) for train, test in cv.split()]
paths = cv.backtest_paths(preds)  # shape (5, len(X)): five out-of-sample backtests
```

A strategy evaluated on every row gives a distribution of out-of-sample Sharpe ratios
rather than the single number a walk-forward test produces.

## References

- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance* 20(4), 39-69.
- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk* 15(2), 3-44.
- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting on out-of-sample performance. *Notices of the American Mathematical Society* 61(5), 458-471.
- Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management* 40(5), 94-107.
- Galwey, N. W. (2009). A new measure of the effective number of tests. *Genetic Epidemiology* 33(7), 559-568.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics* 23(4), 365-380.
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1), 13-28.
- Hsu, P.-H., Hsu, Y.-C. and Kuan, C.-M. (2010). Testing the predictive ability of technical analysis using a new stepwise test without data snooping bias. *Journal of Empirical Finance* 17(3), 471-484.
- Li, J. and Ji, L. (2005). Adjusting multiple testing in multilocus analyses using the eigenvalues of a correlation matrix. *Heredity* 95, 221-227.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4), 36-52.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working paper, University of Basel.
- Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies* 61(4), 631-653.
- Nyholt, D. R. (2004). A simple correction for multiple testing for SNPs in linkage disequilibrium with each other. *American Journal of Human Genetics* 74(4), 765-769.
- Patton, A., Politis, D. N. and White, H. (2009). Correction to "Automatic block-length selection for the dependent bootstrap". *Econometric Reviews* 28(4), 372-375.
- Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association* 89(428), 1303-1313.
- Politis, D. N. and White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews* 23(1), 53-70.
- Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica* 73(4), 1237-1282.
- Romano, J. P. and Wolf, M. (2016). Efficient computation of adjusted p-values for resampling-based stepdown multiple testing. *Statistics & Probability Letters* 113, 38-40.
- Sullivan, R., Timmermann, A. and White, H. (1999). Data-snooping, technical trading rule performance, and the bootstrap. *Journal of Finance* 54(5), 1647-1691.
- White, H. (2000). A reality check for data snooping. *Econometrica* 68(5), 1097-1126.
