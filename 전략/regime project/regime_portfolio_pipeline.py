from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()


def load_env_file(env_path: Optional[Path] = None) -> None:
    path = env_path or BASE_DIR / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


load_env_file()


def get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with "
            "`pip install matplotlib` or `pip install -r requirements.txt`."
        ) from exc

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


FRED_SERIES = {
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "VIXCLS": "VIX",
    "BAA10YM": "Credit_Spread",
    "T10Y2Y": "Yield_Spread",
    "UNRATE": "Unemployment",
    "CPIAUCSL": "CPI",
}

REGIME_LABELS = {
    0: "Calm",
    1: "Risk-Off",
    2: "Inflation Shock",
}


@dataclass
class PipelineConfig:
    fred_api_key: Optional[str] = field(default_factory=lambda: os.getenv("FRED_API_KEY"))
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    etf_tickers: List[str] = field(default_factory=lambda: ["SHY", "IEF", "TLT"])
    etf_duration: Dict[str, float] = field(
        default_factory=lambda: {"SHY": 2.0, "IEF": 7.5, "TLT": 17.5}
    )
    pc1_window_days: int = 63
    hmm_features: List[str] = field(
        default_factory=lambda: [
            "PC1_Var_Z",
            "VIX_Z",
            "Credit_Spread_Z",
            "Yield_Spread_Z",
            "Inflation_YoY_Z",
        ]
    )
    rolling_window_months: int = 84
    n_regimes: int = 3
    covariance_type: str = "diag"
    hmm_n_iter: int = 500
    hmm_n_starts: int = 10
    random_state: int = 42
    min_regime_length: int = 3
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    probability_smoothing_alpha: float = 0.35
    transaction_cost_bps: float = 4.0
    no_trade_zone: float = 0.02
    regime_duration_targets: Dict[int, float] = field(
        default_factory=lambda: {
            0: 8.0,
            1: 14.0,
            2: 3.0,
        }
    )


def fetch_fred_series(
    series_id: str,
    api_key: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    if end_date:
        params["observation_end"] = end_date

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    observations = response.json().get("observations", [])

    frame = pd.DataFrame(observations)
    if frame.empty:
        return pd.Series(dtype=float, name=series_id)

    series = pd.to_numeric(frame["value"], errors="coerce")
    series.index = pd.to_datetime(frame["date"])
    series = series.dropna()
    series.name = series_id
    return series


def fetch_fred_data(config: PipelineConfig) -> pd.DataFrame:
    if not config.fred_api_key:
        raise ValueError(
            "FRED API key is required. Add `FRED_API_KEY=...` to `.env` in this folder "
            "or set the environment variable directly."
        )

    columns = []
    for series_id, column_name in FRED_SERIES.items():
        series = fetch_fred_series(
            series_id=series_id,
            api_key=config.fred_api_key,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        columns.append(series.rename(column_name))

    fred_data = pd.concat(columns, axis=1).sort_index()
    business_index = pd.bdate_range(fred_data.index.min(), fred_data.index.max())
    return fred_data.reindex(business_index).ffill()


def fetch_etf_prices(config: PipelineConfig) -> pd.DataFrame:
    prices = yf.download(
        config.etf_tickers,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    return prices.dropna(how="all")


def rolling_pc1_variance_share(yield_curve: pd.DataFrame, window_days: int) -> pd.Series:
    changes = yield_curve.diff().dropna()
    values = []
    dates = []

    scaler = StandardScaler()
    for end_idx in range(window_days, len(changes) + 1):
        window = changes.iloc[end_idx - window_days : end_idx]
        scaled = scaler.fit_transform(window)
        pca = PCA(n_components=min(window.shape))
        pca.fit(scaled)

        values.append(float(pca.explained_variance_ratio_[0]))
        dates.append(window.index[-1])

    return pd.Series(values, index=pd.DatetimeIndex(dates), name="PC1_Var")


def expanding_zscore(series: pd.Series, min_periods: int = 24) -> pd.Series:
    mean = series.expanding(min_periods=min_periods).mean().shift(1)
    std = series.expanding(min_periods=min_periods).std().shift(1)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def build_monthly_features(fred_data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    yield_curve = fred_data[["6M", "1Y", "2Y", "5Y", "7Y", "10Y"]].copy()
    pc1_var = rolling_pc1_variance_share(yield_curve, window_days=config.pc1_window_days)

    monthly = fred_data.resample("ME").last().copy()
    monthly["Inflation_YoY"] = monthly["CPI"].pct_change(12) * 100.0
    monthly["Unemployment_Change"] = monthly["Unemployment"].diff()

    features = pd.concat(
        [
            pc1_var.resample("ME").last().rename("PC1_Var"),
            monthly[["VIX", "Credit_Spread", "Yield_Spread", "Inflation_YoY", "Unemployment_Change"]],
        ],
        axis=1,
    ).dropna()

    lagged_macro_cols = ["VIX", "Credit_Spread", "Yield_Spread", "Inflation_YoY", "Unemployment_Change"]
    features[lagged_macro_cols] = features[lagged_macro_cols].shift(1)
    features = features.dropna()

    for column in ["PC1_Var", *lagged_macro_cols]:
        features[f"{column}_Z"] = expanding_zscore(features[column], min_periods=24)

    feature_columns = ["PC1_Var_Z", "VIX_Z", "Credit_Spread_Z", "Yield_Spread_Z", "Inflation_YoY_Z"]
    return features.dropna(subset=feature_columns)


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    month_end_prices = prices.resample("ME").last().dropna(how="all")
    return month_end_prices.pct_change().dropna(how="all")


def winsorize_frame(frame: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    clipped = frame.copy()
    for column in clipped.columns:
        lo = clipped[column].quantile(lower)
        hi = clipped[column].quantile(upper)
        clipped[column] = clipped[column].clip(lower=lo, upper=hi)
    return clipped


def fit_best_hmm(
    scaled_values: np.ndarray,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    n_starts: int,
    base_random_state: int,
) -> Tuple[Optional[GaussianHMM], float, bool]:
    best_model = None
    best_score = -np.inf
    best_converged = False

    for offset in range(n_starts):
        model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=base_random_state + offset,
        )

        try:
            model.fit(scaled_values)
            score = model.score(scaled_values)
            converged = bool(getattr(model.monitor_, "converged", False))
        except Exception:
            continue

        if np.isfinite(score) and score > best_score:
            best_model = model
            best_score = score
            best_converged = converged

    return best_model, float(best_score), best_converged


def map_regime_states(model: GaussianHMM, feature_columns: List[str]) -> Tuple[Dict[int, int], Dict[int, str]]:
    means = pd.DataFrame(model.means_, columns=feature_columns)

    calm_state = means["PC1_Var_Z"].idxmin()
    remaining = means.drop(index=calm_state)

    risk_off_score = (
        remaining["Credit_Spread_Z"].rank(method="first")
        - remaining["Yield_Spread_Z"].rank(method="first")
    )
    risk_off_state = int(risk_off_score.idxmax())
    inflation_state = int([state for state in remaining.index if state != risk_off_state][0])

    state_map = {
        int(calm_state): 0,
        risk_off_state: 1,
        inflation_state: 2,
    }
    return state_map, REGIME_LABELS


def smooth_regime_sequence(regime_series: pd.Series, min_regime_length: int) -> pd.Series:
    smoothed = regime_series.dropna().astype(int).copy()
    values = np.array(smoothed.to_numpy(), copy=True)

    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[end + 1] == values[start]:
            end += 1

        run_length = end - start + 1
        if run_length < min_regime_length:
            previous_state = values[start - 1] if start > 0 else np.nan
            next_state = values[end + 1] if end + 1 < len(values) else np.nan

            if not np.isnan(previous_state) and not np.isnan(next_state) and previous_state == next_state:
                values[start : end + 1] = previous_state
            elif not np.isnan(previous_state):
                values[start : end + 1] = previous_state
            elif not np.isnan(next_state):
                values[start : end + 1] = next_state

        start = end + 1

    smoothed[:] = values
    return smoothed.astype(int)


def rolling_hmm_regime_detection(
    features: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    input_frame = winsorize_frame(
        features[config.hmm_features].dropna(),
        lower=config.winsor_lower,
        upper=config.winsor_upper,
    )

    results = []
    for end_idx in range(config.rolling_window_months, len(input_frame) + 1):
        train_window = input_frame.iloc[end_idx - config.rolling_window_months : end_idx].copy()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(train_window)

        model, score, converged = fit_best_hmm(
            scaled_values=scaled,
            n_components=config.n_regimes,
            covariance_type=config.covariance_type,
            n_iter=config.hmm_n_iter,
            n_starts=config.hmm_n_starts,
            base_random_state=config.random_state,
        )

        current_date = train_window.index[-1]
        if model is None:
            row = {
                "Raw_Regime": np.nan,
                "Raw_Regime_Name": np.nan,
                "Confidence": np.nan,
                "Entropy": np.nan,
                "Model_Score": np.nan,
                "Converged": False,
            }
            row.update({f"Prob_State_{state}": np.nan for state in range(config.n_regimes)})
            results.append((current_date, row))
            continue

        hidden_states = model.predict(scaled)
        probabilities = model.predict_proba(scaled)

        state_map, regime_names = map_regime_states(model=model, feature_columns=list(train_window.columns))

        mapped_states = np.array([state_map[state] for state in hidden_states])
        mapped_probabilities = np.zeros_like(probabilities)
        for raw_state, mapped_state in state_map.items():
            mapped_probabilities[:, mapped_state] = probabilities[:, raw_state]

        last_probabilities = mapped_probabilities[-1]
        last_regime = int(mapped_states[-1])

        row = {
            "Raw_Regime": last_regime,
            "Raw_Regime_Name": regime_names[last_regime],
            "Confidence": float(np.max(last_probabilities)),
            "Entropy": float(-np.sum(last_probabilities * np.log(np.clip(last_probabilities, 1e-12, 1.0)))),
            "Model_Score": score,
            "Converged": converged,
        }
        row.update(
            {
                f"Prob_State_{state}": float(last_probabilities[state])
                for state in range(config.n_regimes)
            }
        )
        results.append((current_date, row))

    regime_df = pd.DataFrame(
        [row for _, row in results],
        index=pd.Index([date for date, _ in results], name="Date"),
    )

    smoothed = smooth_regime_sequence(
        regime_df["Raw_Regime"].dropna(),
        min_regime_length=config.min_regime_length,
    )
    regime_df["Smoothed_Regime"] = pd.Series(index=regime_df.index, dtype="Int64")
    regime_df.loc[smoothed.index, "Smoothed_Regime"] = smoothed.astype("Int64")
    regime_df["Smoothed_Regime_Name"] = regime_df["Smoothed_Regime"].map(REGIME_LABELS)
    return regime_df


def smooth_probabilities(regime_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    probability_columns = [column for column in regime_df.columns if column.startswith("Prob_State_")]
    return regime_df[probability_columns].ewm(alpha=alpha, adjust=False).mean()


def target_duration_from_probabilities(
    probability_row: pd.Series,
    regime_duration_targets: Dict[int, float],
) -> float:
    duration = 0.0
    for state, target in regime_duration_targets.items():
        duration += float(probability_row.get(f"Prob_State_{state}", 0.0)) * target
    return duration


def duration_to_weights(target_duration: float, etf_duration: Dict[str, float]) -> pd.Series:
    shy_duration = etf_duration["SHY"]
    ief_duration = etf_duration["IEF"]
    tlt_duration = etf_duration["TLT"]

    weights = pd.Series({"SHY": 0.0, "IEF": 0.0, "TLT": 0.0}, dtype=float)
    clipped_target = float(np.clip(target_duration, shy_duration, tlt_duration))

    if clipped_target <= ief_duration:
        weight_ief = (clipped_target - shy_duration) / (ief_duration - shy_duration)
        weights["IEF"] = weight_ief
        weights["SHY"] = 1.0 - weight_ief
    else:
        weight_tlt = (clipped_target - ief_duration) / (tlt_duration - ief_duration)
        weights["TLT"] = weight_tlt
        weights["IEF"] = 1.0 - weight_tlt

    return weights


def backtest_regime_duration_strategy(
    regime_df: pd.DataFrame,
    etf_returns: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    smoothed_probs = smooth_probabilities(regime_df, alpha=config.probability_smoothing_alpha)
    signal_frame = regime_df.join(smoothed_probs.add_prefix("Smoothed_"), how="inner")
    common_index = signal_frame.index.intersection(etf_returns.index)
    signal_frame = signal_frame.loc[common_index]
    etf_returns = etf_returns.loc[common_index]

    strategy_rows = []
    previous_weights = pd.Series({ticker: 0.0 for ticker in config.etf_tickers}, dtype=float)

    for idx in range(len(signal_frame) - 1):
        signal_date = signal_frame.index[idx]
        next_date = signal_frame.index[idx + 1]

        probability_row = pd.Series(
            {
                f"Prob_State_{state}": signal_frame.iloc[idx][f"Smoothed_Prob_State_{state}"]
                for state in range(config.n_regimes)
            }
        )

        target_duration = target_duration_from_probabilities(
            probability_row=probability_row,
            regime_duration_targets=config.regime_duration_targets,
        )
        target_weights = duration_to_weights(target_duration, config.etf_duration)

        if float(np.abs(target_weights - previous_weights).sum()) < config.no_trade_zone:
            executed_weights = previous_weights.copy()
        else:
            executed_weights = target_weights.copy()

        turnover = float(np.abs(executed_weights - previous_weights).sum())
        tcost = turnover * (config.transaction_cost_bps / 10000.0)
        realized_return = float((executed_weights * etf_returns.loc[next_date]).sum() - tcost)

        strategy_rows.append(
            {
                "Signal_Date": signal_date,
                "Applied_Date": next_date,
                "Regime": signal_frame.iloc[idx]["Smoothed_Regime"],
                "Regime_Name": signal_frame.iloc[idx]["Smoothed_Regime_Name"],
                "Confidence": signal_frame.iloc[idx]["Confidence"],
                "Target_Duration": target_duration,
                "Turnover": turnover,
                "Transaction_Cost": tcost,
                "Portfolio_Return": realized_return,
                "SHY": executed_weights["SHY"],
                "IEF": executed_weights["IEF"],
                "TLT": executed_weights["TLT"],
            }
        )

        previous_weights = executed_weights

    details = pd.DataFrame(strategy_rows).set_index("Applied_Date")
    returns = details["Portfolio_Return"].rename("Regime_Duration_Strategy")
    return returns, details


def benchmark_returns(etf_returns: pd.DataFrame) -> pd.DataFrame:
    benchmarks = pd.DataFrame(index=etf_returns.index)
    benchmarks["IEF_Only"] = etf_returns["IEF"]
    benchmarks["Equal_Weight"] = etf_returns[["SHY", "IEF", "TLT"]].mean(axis=1)
    return benchmarks


def performance_metrics(returns: pd.Series, periods_per_year: int = 12) -> Dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {}

    cumulative = (1.0 + clean).cumprod()
    total_periods = len(clean)

    cagr = cumulative.iloc[-1] ** (periods_per_year / total_periods) - 1.0
    volatility = clean.std() * math.sqrt(periods_per_year)
    downside = clean.clip(upper=0).std() * math.sqrt(periods_per_year)
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0

    return {
        "CAGR": float(cagr),
        "Volatility": float(volatility),
        "Sharpe": float(cagr / volatility) if volatility > 0 else np.nan,
        "Sortino": float(cagr / downside) if downside > 0 else np.nan,
        "Max_Drawdown": float(drawdown.min()),
        "Hit_Ratio": float((clean > 0).mean()),
    }


def summarize_performance(return_frame: pd.DataFrame) -> pd.DataFrame:
    summary = {
        column: performance_metrics(return_frame[column])
        for column in return_frame.columns
    }
    return pd.DataFrame(summary).T.sort_values("Sharpe", ascending=False)


def regime_feature_summary(features: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    merged = features.join(
        regime_df[["Smoothed_Regime", "Smoothed_Regime_Name"]],
        how="inner",
    ).dropna(subset=["Smoothed_Regime"])

    summary = (
        merged.groupby(["Smoothed_Regime", "Smoothed_Regime_Name"])[
            ["PC1_Var", "VIX", "Credit_Spread", "Yield_Spread", "Inflation_YoY"]
        ]
        .agg(["mean", "median", "std", "count"])
    )
    return summary


def regime_performance_summary(strategy_details: pd.DataFrame) -> pd.DataFrame:
    grouped = strategy_details.groupby("Regime_Name")["Portfolio_Return"]
    summary = pd.DataFrame(
        {
            "Avg_Monthly_Return": grouped.mean(),
            "Volatility": grouped.std(),
            "Hit_Ratio": grouped.apply(lambda x: (x > 0).mean()),
            "Observations": grouped.count(),
        }
    )
    return summary.sort_index()


def plot_regime_overview(features: pd.DataFrame, regime_df: pd.DataFrame) -> None:
    plt = get_plt()
    plot_data = features[["PC1_Var"]].join(
        regime_df[["Smoothed_Regime", "Smoothed_Regime_Name"]],
        how="inner",
    )

    colors = {
        0: "#d8ecff",
        1: "#ffd8d8",
        2: "#ffe7b8",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_data.index, plot_data["PC1_Var"], color="black", linewidth=1.8, label="PC1 Variance Share")

    for regime, color in colors.items():
        mask = plot_data["Smoothed_Regime"] == regime
        ax.fill_between(
            plot_data.index,
            0,
            1,
            where=mask,
            color=color,
            alpha=0.45,
            transform=ax.get_xaxis_transform(),
            label=REGIME_LABELS[regime],
        )

    ax.set_title("Regime Detection Overview")
    ax.set_ylabel("PC1 Variance Share")
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


def plot_regime_feature_boxplots(features: pd.DataFrame, regime_df: pd.DataFrame) -> None:
    plt = get_plt()
    plot_data = features.join(
        regime_df[["Smoothed_Regime_Name"]],
        how="inner",
    ).dropna(subset=["Smoothed_Regime_Name"])

    feature_columns = ["PC1_Var", "VIX", "Credit_Spread", "Yield_Spread", "Inflation_YoY"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, column in enumerate(feature_columns):
        ax = axes[idx]
        groups = [
            plot_data.loc[plot_data["Smoothed_Regime_Name"] == label, column].dropna()
            for label in REGIME_LABELS.values()
        ]
        ax.boxplot(groups, labels=list(REGIME_LABELS.values()), patch_artist=True)
        ax.set_title(column)
        ax.tick_params(axis="x", rotation=15)

    axes[-1].axis("off")
    fig.suptitle("Feature Distribution by Regime", fontsize=14)
    plt.tight_layout()
    return fig


def plot_regime_probabilities(regime_df: pd.DataFrame) -> None:
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(14, 5))
    for state, label in REGIME_LABELS.items():
        ax.plot(regime_df.index, regime_df[f"Prob_State_{state}"], linewidth=1.6, label=label)

    ax.set_title("Posterior Regime Probabilities")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


def plot_strategy_weights(strategy_details: pd.DataFrame) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    strategy_details[["SHY", "IEF", "TLT"]].plot.area(
        ax=axes[0],
        stacked=True,
        alpha=0.85,
        color=["#7db7ff", "#4f7cac", "#1d3557"],
    )
    axes[0].set_title("Portfolio Weights Through Time")
    axes[0].set_ylabel("Weight")
    axes[0].legend(loc="upper left", ncol=3)

    strategy_details["Target_Duration"].plot(ax=axes[1], color="#c44536", linewidth=2)
    axes[1].set_title("Target Duration")
    axes[1].set_ylabel("Years")
    axes[1].set_xlabel("")
    plt.tight_layout()
    return fig


def plot_backtest_dashboard(return_frame: pd.DataFrame) -> None:
    plt = get_plt()
    cumulative = (1.0 + return_frame).cumprod()
    drawdown = cumulative.div(cumulative.cummax()).sub(1.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    cumulative.plot(ax=axes[0], linewidth=1.8)
    axes[0].set_title("Cumulative Performance")
    axes[0].set_ylabel("Growth of $1")

    drawdown.plot(ax=axes[1], linewidth=1.5)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    plt.tight_layout()
    return fig


def plot_regime_performance(strategy_details: pd.DataFrame) -> None:
    plt = get_plt()
    summary = regime_performance_summary(strategy_details)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    summary["Avg_Monthly_Return"].plot.bar(ax=axes[0], color=["#7cc6fe", "#f4a261", "#e76f51"])
    axes[0].set_title("Average Monthly Return by Regime")
    axes[0].set_ylabel("Return")
    axes[0].tick_params(axis="x", rotation=15)

    summary["Hit_Ratio"].plot.bar(ax=axes[1], color=["#7cc6fe", "#f4a261", "#e76f51"])
    axes[1].set_title("Hit Ratio by Regime")
    axes[1].set_ylabel("Hit Ratio")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    return fig


def plot_backtest(return_frame: pd.DataFrame) -> None:
    plt = get_plt()
    cumulative = (1.0 + return_frame).cumprod()
    fig, ax = plt.subplots(figsize=(14, 6))
    cumulative.plot(ax=ax, linewidth=1.8)
    ax.set_title("Strategy vs Benchmarks")
    ax.set_ylabel("Cumulative Return")
    plt.tight_layout()
    return fig


def export_readme_figures(
    results: Dict[str, pd.DataFrame | pd.Series],
    output_dir: str = "figures",
) -> Dict[str, str]:
    plt = get_plt()
    output_path = BASE_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {
        "regime_overview.png": plot_regime_overview(results["features"], results["regimes"]),
        "regime_probabilities.png": plot_regime_probabilities(results["regimes"]),
        "regime_features.png": plot_regime_feature_boxplots(results["features"], results["regimes"]),
        "strategy_weights.png": plot_strategy_weights(results["strategy_details"]),
        "regime_performance.png": plot_regime_performance(results["strategy_details"]),
        "backtest_dashboard.png": plot_backtest_dashboard(results["returns"]),
    }

    saved = {}
    for filename, figure in figures.items():
        figure.savefig(output_path / filename, dpi=180, bbox_inches="tight")
        saved[filename] = str(output_path / filename)
        plt.close(figure)

    return saved


def run_pipeline(config: PipelineConfig) -> Dict[str, pd.DataFrame | pd.Series]:
    fred_data = fetch_fred_data(config)
    etf_prices = fetch_etf_prices(config)
    features = build_monthly_features(fred_data, config)
    regime_df = rolling_hmm_regime_detection(features, config)
    etf_rets = monthly_returns(etf_prices)

    strategy_returns, strategy_details = backtest_regime_duration_strategy(
        regime_df=regime_df,
        etf_returns=etf_rets,
        config=config,
    )

    benchmarks = benchmark_returns(etf_rets)
    comparison = pd.concat([strategy_returns, benchmarks], axis=1).dropna()
    metrics = summarize_performance(comparison)

    return {
        "fred_data": fred_data,
        "etf_prices": etf_prices,
        "features": features,
        "regimes": regime_df,
        "strategy_details": strategy_details,
        "returns": comparison,
        "metrics": metrics,
    }


def main() -> None:
    plt = get_plt()
    config = PipelineConfig()
    results = run_pipeline(config)

    print(results["metrics"].round(4))
    print(regime_performance_summary(results["strategy_details"]).round(4))
    plot_regime_overview(results["features"], results["regimes"])
    plt.show()
    plot_regime_probabilities(results["regimes"])
    plt.show()
    plot_regime_feature_boxplots(results["features"], results["regimes"])
    plt.show()
    plot_strategy_weights(results["strategy_details"])
    plt.show()
    plot_regime_performance(results["strategy_details"])
    plt.show()
    plot_backtest_dashboard(results["returns"])
    plt.show()
    plot_backtest(results["returns"])
    plt.show()


if __name__ == "__main__":
    main()
