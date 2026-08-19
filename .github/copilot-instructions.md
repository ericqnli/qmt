# QMT Strategy Repository Instructions

## Runtime and validation

- This repository contains two standalone Python strategies for the Guojin QMT/XtQuant model runtime; it is not a Python package. QMT calls each script's `init(C)` once and `handlebar(C)` for every bar.
- Target Guojin full QMT (standard/full-feature edition), not miniQMT. Bind the strategy account during `init` with `C.set_account(account_id)`; without it, order and deal callbacks do not fire.
- Configure the QMT model period as daily (`1d`) and validate strategy changes through a QMT backtest. There are no repository-native build, test, lint, dependency, or single-test commands.
- `Macd_V1.py` requires NumPy and TA-Lib. Both scripts rely on the QMT-provided context/API (`C`, market-data methods, position methods, `passorder`, and `download_history_data`) rather than locally defined adapters.
- The tracked Python files are UTF-8 despite their `# -*- coding: gbk -*-` headers. Do not rely on a default `python -m py_compile` check until that pre-existing declaration mismatch is resolved.

## Architecture

- `Macd_V1.py` implements a MACD trend filter with regime-adaptive KDJ/RSI oversold entries. Module constants form the user configuration; `init(C)` copies them into the QMT context and initializes state. `_process_one` fetches daily OHLCV data, calculates TA-Lib indicators, reconciles state with broker positions, prioritizes exits, then checks entries and submits or notifies according to `TRADE_MODE`.
- `日线_RSRS统一策略_开关版.py` implements the long-only RSRS strategy. Its `init(C)` configures the `C.cfg` signal dictionary and state; `handlebar(C)` obtains daily high/low/close (and amount when enabled), computes the RSRS signal, runs the binary `next_position` state machine, and maps transitions to `do_buy` or `do_sell_all`.
- Both strategies keep in-memory per-symbol state on `C.pos_state`, deduplicate daily signals with `C.signal_seen`, and use the last closed bar for live decisions (`-2`, with `-3` as the prior bar) to avoid intraday signals. Preserve this indexing behavior when changing indicators or execution logic.

## Repository conventions

- Keep strategy configuration at the top-level configuration area in `Macd_V1.py` or in `init(C)` for the RSRS script; use the `C` context for runtime state and values consumed by helpers.
- Treat order submission as safety-sensitive. `Macd_V1.py` has `backtest`, `notify`, and `auto` modes; `notify` sends a WeCom webhook and does not submit orders. RSRS defaults to `C.enable_order = False`. Do not enable live orders or add real account numbers/webhook URLs in committed code.
- Use native full-QMT `deal_callback` and `order_callback` callbacks rather than polling for trade notifications. QMT direction codes are `48` for buy and `49` for sell; status `57` means rejected/junk and should produce an additional `@all` WeCom alert.
- Enterprise WeChat group-robot webhooks are the notification target. Keep the standard-library `urllib` and `json` implementation; do not add `requests` unless explicitly requested.
- QMT APIs return varying data/position shapes. Preserve the existing compatibility and fallback handling (`get_market_data_ex`/`get_market_data`/`get_history_data`, numeric/object/dict positions) when modifying data access or reconciliation.
- Order quantities must remain whole 100-share lots. Maintain signal deduplication and only update locally managed positions after a successful submission path, except for the RSRS target-position intent that is explicitly recorded after a state transition.
- Keep the existing Chinese strategy comments and log messages intact when their meaning remains correct; they document operational behavior used in QMT backtests and live monitoring.
