# Regime-Based Duration Rotation

This folder contains a cleaned research pipeline for a bond regime strategy.

## Files

- `regime_portfolio_pipeline.py`: reusable end-to-end pipeline from data download to regime detection and backtest.
- `regime_portfolio_pipeline.ipynb`: portfolio-ready notebook version for GitHub presentation.
- Existing legacy files are kept for reference and were not overwritten.

## Strategy Flow

1. Download U.S. Treasury curve, macro variables, and bond ETF prices.
2. Build a monthly feature set using rolling PCA on daily yield changes.
3. Detect 3 macro/rate regimes with a rolling HMM.
4. Convert regime probabilities into a target duration.
5. Map target duration to `SHY`, `IEF`, `TLT` weights and backtest.

## Required Packages

```bash
pip install pandas numpy matplotlib requests yfinance scikit-learn hmmlearn
```

## Environment

Fill in [`.env`](/Users/2ys/Desktop/KSIF/SAA/전략/regime%20project/.env) in this folder:

```bash
FRED_API_KEY=your_key_here
```

You can also copy [`.env.example`](/Users/2ys/Desktop/KSIF/SAA/전략/regime%20project/.env.example) if needed.

## Run

```bash
python regime_portfolio_pipeline.py
```

Or open `regime_portfolio_pipeline.ipynb` and run the cells sequentially.
