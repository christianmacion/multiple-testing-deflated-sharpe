"""
Project 1 — Multiple Testing & the Deflated Sharpe Ratio
========================================================
Hypothesis-free by design: this project demonstrates *why a single headline
Sharpe is not evidence*. We mine a large family of moving-average crossover
rules on BTC, take the best in-sample Sharpe, and show it (a) is roughly what
PURE NOISE of the same trial count would produce (False Strategy Theorem),
(b) collapses out-of-sample, and (c) is rejected by the Deflated Sharpe Ratio.

Free data: Binance BTCUSDT daily. numpy/pandas/matplotlib + quantlib.
Author: Christian Macion.
"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantlib"))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import quantlib as q

COST_BPS = 6.0      # realistic crypto taker round-turn, in bps of notional per unit turnover
ANN = 365           # crypto trades every day

def backtest_ma(close, fast, slow, long_short=True):
    """Long when fast MA > slow MA; (optionally) short otherwise. Trade next bar."""
    f = close.rolling(fast).mean(); s = close.rolling(slow).mean()
    pos = np.where(f > s, 1.0, -1.0 if long_short else 0.0)
    pos = pd.Series(pos, index=close.index).shift(1)           # .shift(1): no look-ahead
    ret = close.pct_change()
    turnover = pos.diff().abs().fillna(0.0)
    net = pos * ret - turnover * COST_BPS * 1e-4
    return net.dropna()

def main():
    print("Pulling BTC daily history from Binance ...")
    btc = q.binance_klines("BTCUSDT", "1d", start="2017-08-01")
    close = btc["close"]
    n = len(close); split = int(n * 0.6)
    is_close, oos_close = close.iloc[:split], close.iloc[split:]
    print(f"  {n} daily bars  |  IS {is_close.index[0].date()}..{is_close.index[-1].date()}"
          f"  OOS {oos_close.index[0].date()}..{oos_close.index[-1].date()}")

    fasts = [5, 8, 10, 15, 20, 25, 30, 40, 50]
    slows = [50, 75, 100, 125, 150, 175, 200, 225, 250]
    rows = []
    for f in fasts:
        for s in slows:
            if f >= s: continue
            for ls in (True, False):
                is_net = backtest_ma(is_close, f, s, ls)
                oos_net = backtest_ma(oos_close, f, s, ls)
                rows.append({"fast": f, "slow": s, "ls": ls,
                             "is_sharpe": q.sharpe(is_net, ANN),
                             "oos_sharpe": q.sharpe(oos_net, ANN),
                             "is_net": is_net})
    res = pd.DataFrame(rows)
    N = len(res)
    best = res.loc[res["is_sharpe"].idxmax()]
    print(f"\nMined N = {N} strategy variants.")
    print(f"  BEST in-sample:  fast={best.fast} slow={best.slow} "
          f"{'L/S' if best.ls else 'long-only'}  IS Sharpe = {best.is_sharpe:.2f}")
    print(f"  Same variant out-of-sample:  OOS Sharpe = {best.oos_sharpe:.2f}")

    # False Strategy Theorem: E[max Sharpe] of N independent zero-skill strategies
    sharpe_sd = res["is_sharpe"].std(ddof=1)        # dispersion of trial Sharpes (per-year units)
    z1 = q.norm_ppf(1 - 1.0/N); z2 = q.norm_ppf(1 - 1.0/(N*math.e))
    exp_max_noise = sharpe_sd * ((1 - q.EULER_GAMMA)*z1 + q.EULER_GAMMA*z2)
    print(f"\nFalse Strategy Theorem (N={N}, trial-Sharpe sd={sharpe_sd:.2f}):")
    print(f"  Expected max Sharpe from PURE NOISE of this search ≈ {exp_max_noise:.2f}")
    print(f"  Observed best IS Sharpe = {best.is_sharpe:.2f}  → ratio "
          f"{best.is_sharpe/exp_max_noise:.2f}× the noise expectation")

    # Deflated Sharpe Ratio on the winner (per-trade Sharpes feed the variance input)
    trial_sharpes = res["is_sharpe"].values / math.sqrt(ANN)   # back to per-period units
    best_daily = best["is_net"]
    dsr = q.deflated_sharpe(best_daily.values, m_effective=N, trial_sharpes=trial_sharpes)
    print(f"\nDeflated Sharpe Ratio of the in-sample winner: DSR = {dsr:.3f}  "
          f"(advisory pass ≥ 0.95 → {'PASS' if dsr>=0.95 else 'FAIL — not distinguishable from luck'})")

    # IS vs OOS correlation across the whole family (does ranking persist?)
    rank_corr = np.corrcoef(res["is_sharpe"].rank(), res["oos_sharpe"].rank())[0, 1]  # Spearman = Pearson on ranks (no scipy)
    print(f"\nIS→OOS Spearman rank correlation across the {N} variants = {rank_corr:.2f}  "
          f"(near 0 ⇒ in-sample ranking does NOT predict out-of-sample)")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(res["is_sharpe"], bins=25, color="#3b5b8c", alpha=0.85)
    ax[0].axvline(exp_max_noise, color="crimson", ls="--", lw=2, label=f"E[max | noise] ≈ {exp_max_noise:.2f}")
    ax[0].axvline(best.is_sharpe, color="black", lw=2, label=f"observed best = {best.is_sharpe:.2f}")
    ax[0].set_title(f"In-sample Sharpe of N={N} mined variants"); ax[0].set_xlabel("annualized Sharpe"); ax[0].legend(fontsize=8)
    ax[1].scatter(res["is_sharpe"], res["oos_sharpe"], s=14, alpha=0.6, color="#3b5b8c")
    ax[1].axhline(0, color="gray", lw=0.7); ax[1].axvline(0, color="gray", lw=0.7)
    ax[1].scatter([best.is_sharpe], [best.oos_sharpe], color="black", s=60, zorder=5, label="IS winner")
    ax[1].set_title(f"IS vs OOS Sharpe (rank corr = {rank_corr:.2f})"); ax[1].set_xlabel("in-sample Sharpe"); ax[1].set_ylabel("out-of-sample Sharpe"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "results", "figure.png"), dpi=130)

    out = {"N_trials": N, "best_fast": int(best.fast), "best_slow": int(best.slow),
           "best_is_sharpe": round(float(best.is_sharpe), 3), "best_oos_sharpe": round(float(best.oos_sharpe), 3),
           "expected_max_noise_sharpe": round(float(exp_max_noise), 3),
           "deflated_sharpe": round(float(dsr), 3), "is_oos_rank_corr": round(float(rank_corr), 3)}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results", "results.json"), "w"), indent=2)
    print("\nSaved results/figure.png and results/results.json")

if __name__ == "__main__":
    main()
