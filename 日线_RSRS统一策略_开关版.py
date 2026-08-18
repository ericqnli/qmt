# -*- coding: gbk -*-
# =============================================================================
# RSRS 统一策略（开关控制版）—— 国金证券 QMT / 迅投 Python 模型
# 基于附件：RSRS_SI.md（主题 ETF 日线择时）
# 【必须】模型周期设为【日线 / 1d】
# =============================================================================
#
# 【一句话】
#   用最近 N 日 high~low 回归得 β，标准化并可选质量/波动/成交额加权修正，
#   得到信号 x；x > +S 开多，x < -S 平仓，中间持有。只做多。
#
# 【主干四档（tier 互斥）】
#   slope    : x = β
#   zscore   : x = z(β)
#   revise   : x = z(β) * R²          ← 默认推荐（预设A）
#   positive : x = z(β) * R² * β
#
# 【加强开关（可叠加在 z 系档）】
#   use_amount_weight : 成交额加权 WLS 回归
#   use_dampen        : 波动分位钝化  x = z * R^(4*q)
#   use_ma_filter     : 开仓需 MA20 上行（默认关）
#   use_vol_corr_filter: 开仓需量与信号近10日相关>0（默认关）
#
# 【推荐预设】
#   预设A（跑通）: tier=revise, amount_w=False, dampen=False, S=0.9
#   预设B（正式）: tier=revise, amount_w=True,  dampen=True,  S=0.9~1.0
#
# 【执行纪律】
#   信号在 T 日收盘计算 → T+1 执行
#   实盘同一时期只冻结一组 config；五只主题 ETF 共用同一 config
#   优先只调 S；少同时改 N/M/tier/加强
#
# 【使用说明】
#   1. 回测/只看信号：C.enable_order = False（默认）
#   2. 模拟/实盘下单：C.enable_order = True，并填写 C.account
#   3. 切换预设：在 init 里改 C.preset = 'A' 或 'B'，或直接改开关
#   4. 日志：改 C.log_level 与各 C.log_mod_xxx
# =============================================================================

import os
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# 日志系统
# 级别：0=OFF  1=ERROR  2=WARN  3=INFO(默认，信号/成交)  4=DETAIL(每bar摘要)
#       5=DEBUG(阈值计算、未触发原因等最全信息)
# ---------------------------------------------------------------------------
LOG_OFF, LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DETAIL, LOG_DEBUG = 0, 1, 2, 3, 4, 5
_LEVEL_NAME = {0: 'OFF', 1: 'ERROR', 2: 'WARN', 3: 'INFO', 4: 'DETAIL', 5: 'DEBUG'}


def _log(C, level, module, msg):
    if not getattr(C, 'log_enable', True):
        return
    if level > getattr(C, 'log_level', LOG_INFO):
        return
    mod = (module or 'SYS').upper()
    switches = {
        'BAR': 'log_mod_bar',
        'POS': 'log_mod_pos',
        'SIGNAL': 'log_mod_signal',
        'DECISION': 'log_mod_decision',
        'ORDER': 'log_mod_order',
        'SKIP': 'log_mod_skip',
        'STAT': 'log_mod_stat',
        'SYS': None, 'ERROR': None, 'WARN': None,
    }
    sw = switches.get(mod)
    if sw is not None and not getattr(C, sw, True):
        return
    if mod == 'BAR' and level >= LOG_DETAIL:
        every = int(getattr(C, 'log_bar_every_n', 1) or 1)
        if every > 1 and int(getattr(C, '_bar_callback_count', 0)) % every != 0:
            return
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}][{_LEVEL_NAME.get(level, level)}][{mod}] {msg}"
    if getattr(C, 'log_to_console', True):
        try:
            print(line)
        except Exception:
            try:
                print(line.encode('gbk', errors='ignore').decode('gbk', errors='ignore'))
            except Exception:
                pass
    if getattr(C, 'log_to_file', False):
        path = getattr(C, 'log_file_path', '') or ''
        if path:
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(line + '\n')
            except Exception as e:
                if not getattr(C, '_log_file_error_reported', False):
                    C._log_file_error_reported = True
                    print(f"[日志] 写文件失败: {e}")


def _log_error(C, msg):
    _log(C, LOG_ERROR, 'ERROR', msg)


def _log_warn(C, msg):
    _log(C, LOG_WARN, 'WARN', msg)


def _inc(C, key, n=1):
    if not hasattr(C, 'stats') or C.stats is None:
        C.stats = {}
    C.stats[key] = int(C.stats.get(key, 0)) + n


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _parse_bar_time(C):
    """解析当前 bar 时间（兼容秒/毫秒）。"""
    try:
        bar_time = C.get_bar_timetag(C.barpos)
        if bar_time > 1e12:
            dt = datetime.fromtimestamp(bar_time / 1000.0)
        else:
            dt = datetime.fromtimestamp(bar_time)
        return dt, dt.strftime('%Y-%m-%d')
    except Exception:
        return None, "----"


def _is_realtime_unclosed_bar(C):
    """实盘最新未收盘 K 线判断。回测中间 bar 一般 False。"""
    try:
        if hasattr(C, 'is_last_bar') and callable(C.is_last_bar):
            return bool(C.is_last_bar())
        if hasattr(C, 'is_last_bar'):
            return bool(C.is_last_bar)
    except Exception:
        pass
    return False


def _signal_index(C):
    """
    决策用下标（日线）：
      - 回测/已收盘：-1 / -2
      - 实盘未收盘最新 bar：-2 / -3（用上一根已收盘日线，避免盘中假信号）
    返回 (决策索引, 前一根索引)
    """
    if _is_realtime_unclosed_bar(C):
        return -2, -3
    return -1, -2


def _lot_round(vol):
    """A股/场内ETF按100股整手向下取整。"""
    return int(vol // 100) * 100


def _calc_vol(amount, price):
    if price is None or price <= 0:
        return 0
    return _lot_round(amount / price)


def _get_broker_position(C, stock):
    pos = 0
    try:
        if not hasattr(C, 'get_position'):
            return 0
        obj = C.get_position(stock)
        if isinstance(obj, (int, float)):
            pos = int(obj)
        elif hasattr(obj, 'm_nVolume'):
            pos = int(obj.m_nVolume)
        elif isinstance(obj, dict):
            pos = int(obj.get('volume', obj.get('m_nVolume', 0)) or 0)
    except Exception as e:
        _log(C, LOG_DEBUG, 'POS', f"{stock} get_position 异常: {e}")
    return pos


def _already_signaled(C, stock, time_str, side_tag):
    """同日同方向同标签只触发一次。"""
    key = (stock, time_str, side_tag)
    if key in C.signal_seen:
        _log(C, LOG_DEBUG, 'SIGNAL', f"{stock} {time_str} {side_tag} 去重跳过")
        _inc(C, 'signal_dedup')
        return True
    C.signal_seen.add(key)
    if len(C.signal_seen) > 8000:
        C.signal_seen = set(list(C.signal_seen)[-3000:])
    return False


def _ensure_pos(C, stock):
    """
    自管持仓状态：
      vol              当前持仓股数
      cost             成本价
      bars_held        持有日线根数
      high_since_entry 入场后最高收盘
      last_bar_key     防重复累计
      target_pos       策略目标仓位 0/1（状态机输出）
    """
    if stock not in C.pos_state:
        C.pos_state[stock] = {
            'vol': 0,
            'cost': 0.0,
            'bars_held': 0,
            'high_since_entry': 0.0,
            'last_bar_key': '',
            'target_pos': 0,
        }
    return C.pos_state[stock]


def _pos_snap(st, price=None):
    cost = float(st.get('cost') or 0)
    pnl = ((price / cost) - 1.0) * 100 if cost > 0 and price else None
    pnl_s = f"{pnl:+.2f}%" if pnl is not None else 'N/A'
    return (
        f"vol={st.get('vol', 0)} cost={cost:.3f} pnl={pnl_s} "
        f"target={st.get('target_pos', 0)} held={st.get('bars_held', 0)}"
    )


# ---------------------------------------------------------------------------
# RSRS 核心计算（严格对应 RSRS_SI.md 第4节伪代码）
# ---------------------------------------------------------------------------

def _ols_beta_r2(x, y):
    """普通最小二乘：y ~ x，返回 (beta, R2)。x=low, y=high。"""
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = np.mean(x)
    ym = np.mean(y)
    ssxx = np.sum((x - xm) ** 2)
    if ssxx < 1e-12:
        return np.nan, np.nan
    beta = np.sum((x - xm) * (y - ym)) / ssxx
    yhat = ym + beta * (x - xm)
    ssres = np.sum((y - yhat) ** 2)
    sstot = np.sum((y - ym) ** 2)
    r2 = 1.0 - ssres / sstot if sstot > 1e-12 else 0.0
    return float(beta), float(max(r2, 1e-6))


def _wls_beta_r2(x, y, w):
    """成交额加权最小二乘。w 为正权重（可未归一）。"""
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = np.sum(w)
    if sw <= 0:
        return np.nan, np.nan
    xm = np.sum(w * x) / sw
    ym = np.sum(w * y) / sw
    ssxx = np.sum(w * (x - xm) ** 2)
    if ssxx < 1e-12:
        return np.nan, np.nan
    beta = np.sum(w * (x - xm) * (y - ym)) / ssxx
    yhat = ym + beta * (x - xm)
    ssres = np.sum(w * (y - yhat) ** 2)
    sstot = np.sum(w * (y - ym) ** 2)
    r2 = 1.0 - ssres / sstot if sstot > 1e-12 else 0.0
    return float(beta), float(max(r2, 1e-6))


def _percentile_rank(val, hist):
    """当前值在历史中的分位 q ∈ (0,1]，最大≈1.0。"""
    hist = np.asarray(hist, dtype=float)
    if len(hist) == 0 or np.isnan(val):
        return np.nan
    return float(np.sum(hist <= val) / len(hist))


def _compute_beta_series(high, low, amount, N, use_amount_weight):
    """
    滑动窗口计算 β 与 R² 序列。
    返回 beta_arr, r2_arr，长度与 high 相同，前 N-1 为 nan。
    """
    L = len(high)
    beta_arr = np.full(L, np.nan)
    r2_arr = np.full(L, np.nan)
    if L < N:
        return beta_arr, r2_arr

    for i in range(N - 1, L):
        h_win = high[i - N + 1: i + 1]
        l_win = low[i - N + 1: i + 1]
        if use_amount_weight and amount is not None:
            a_win = amount[i - N + 1: i + 1]
            if np.sum(a_win) <= 0:
                beta, r2 = _ols_beta_r2(l_win, h_win)
            else:
                beta, r2 = _wls_beta_r2(l_win, h_win, a_win)
        else:
            beta, r2 = _ols_beta_r2(l_win, h_win)
        beta_arr[i] = beta
        r2_arr[i] = r2
    return beta_arr, r2_arr


def _vol_quantile_series(ret, N, M, t):
    """
    计算 t 时刻 N 日波动在过去 M 个波动中的分位 q。
    ret 为收益率序列，索引对齐。
    """
    if t < N - 1:
        return np.nan
    # 当前 vol
    vol_t = np.std(ret[t - N + 1: t + 1], ddof=0)
    # 历史 vol 窗口：从 max(N-1, t-M+1) 到 t
    start = max(N - 1, t - M + 1)
    if start > t:
        return np.nan
    hist_vols = []
    for i in range(start, t + 1):
        hist_vols.append(np.std(ret[i - N + 1: i + 1], ddof=0))
    return _percentile_rank(vol_t, hist_vols)


def compute_rsrs_signal(cfg, high, low, close, amount, t):
    """
    计算 t 时刻最终信号 x。
    严格对应 MD 第4节 compute_signal。
    返回 (x, beta, r2, z, q) 便于日志；不足时 x=nan。
    """
    N = cfg['N']
    M = cfg['M']
    L = len(high)
    if t < N - 1 or t >= L:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # 1. 回归得到当前 β, R²
    h_win = high[t - N + 1: t + 1]
    l_win = low[t - N + 1: t + 1]
    if cfg['use_amount_weight'] and amount is not None:
        a_win = amount[t - N + 1: t + 1]
        if np.sum(a_win) > 0:
            beta, r2 = _wls_beta_r2(l_win, h_win, a_win)
        else:
            beta, r2 = _ols_beta_r2(l_win, h_win)
    else:
        beta, r2 = _ols_beta_r2(l_win, h_win)

    if np.isnan(beta):
        return np.nan, beta, r2, np.nan, np.nan

    # 2. slope 且无钝化 → 直接返回 β
    if cfg['tier'] == 'slope' and not cfg['use_dampen']:
        return beta, beta, r2, np.nan, np.nan

    # 3. 需要 zscore：先算整段 beta 序列（或至少最近 M 个）
    #    为效率，只在需要时计算最近 M+N 段
    need_start = max(0, t - M - N + 2)
    beta_series, r2_series = _compute_beta_series(
        high[need_start: t + 1],
        low[need_start: t + 1],
        amount[need_start: t + 1] if amount is not None else None,
        N,
        cfg['use_amount_weight']
    )
    # 对齐到全局索引：beta_series 对应 need_start ... t
    valid_betas = beta_series[~np.isnan(beta_series)]
    if len(valid_betas) < max(30, M // 10):  # 至少有一定有效点
        # 若历史严重不足，仍尝试用现有计算 z
        pass
    if len(valid_betas) < 2:
        return np.nan, beta, r2, np.nan, np.nan

    # 取最近 min(M, len) 个有效 β 做 z
    window = valid_betas[-M:] if len(valid_betas) >= M else valid_betas
    mu = np.mean(window)
    sigma = np.std(window, ddof=0)
    if sigma < 1e-12:
        return np.nan, beta, r2, np.nan, np.nan
    z = (beta - mu) / sigma

    # 4. 钝化 or 原始档位
    q = np.nan
    if cfg['use_dampen']:
        if cfg['tier'] == 'slope':
            # 不推荐，强制忽略钝化
            x = beta
        else:
            # 需要收益率序列
            ret = np.full(L, np.nan)
            ret[1:] = close[1:] / close[:-1] - 1.0
            q = _vol_quantile_series(ret, N, M, t)
            if np.isnan(q):
                x = z * r2  # 降级
            else:
                R = max(r2, 1e-6)
                x = z * (R ** (4.0 * q))
                if cfg['tier'] == 'positive':
                    x = x * beta
    else:
        # 原始四档
        if cfg['tier'] == 'slope':
            x = beta
        elif cfg['tier'] == 'zscore':
            x = z
        elif cfg['tier'] == 'revise':
            x = z * r2
        elif cfg['tier'] == 'positive':
            x = z * r2 * beta
        else:
            x = z * r2  # fallback

    return float(x), float(beta), float(r2), float(z), float(q) if not np.isnan(q) else np.nan


def next_position(pos, x, S, cfg, ma20=None, ma20_prev3=None, vol_corr=None):
    """
    唯一状态机（MD 第5节）。
    返回 new_pos ∈ {0, 1}
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pos  # 保持

    allow_entry = True
    if cfg.get('use_ma_filter', False) and ma20 is not None and ma20_prev3 is not None:
        allow_entry = allow_entry and (ma20 > ma20_prev3)
    if cfg.get('use_vol_corr_filter', False) and vol_corr is not None:
        allow_entry = allow_entry and (vol_corr > 0)

    if pos == 0:
        if x > S and allow_entry:
            return 1
        return 0
    else:  # pos == 1
        if x < -S:
            return 0
        return 1


# ---------------------------------------------------------------------------
# 下单封装
# ---------------------------------------------------------------------------

def _passorder_buy(C, stock, vol):
    if not C.enable_order:
        _log(C, LOG_INFO, 'ORDER', f"{stock} [模拟] 买入 vol={vol}（enable_order=False）")
        return True
    try:
        # 23=买入, 1101=普通委托, 14=最新价?, -1=市价类
        C.passorder(23, 1101, C.account, stock, 14, -1, vol, 'RSRS买', 1, '', C)
        _log(C, LOG_INFO, 'ORDER', f"{stock} passorder买入 account={C.account} vol={vol}")
        _inc(C, 'order_buy_ok')
        return True
    except Exception as e:
        _log_error(C, f"{stock} 买入下单失败: {e}")
        _inc(C, 'order_buy_fail')
        return False


def _passorder_sell(C, stock, vol):
    if not C.enable_order:
        _log(C, LOG_INFO, 'ORDER', f"{stock} [模拟] 卖出 vol={vol}（enable_order=False）")
        return True
    try:
        C.passorder(24, 1101, C.account, stock, 14, -1, vol, 'RSRS卖', 1, '', C)
        _log(C, LOG_INFO, 'ORDER', f"{stock} passorder卖出 account={C.account} vol={vol}")
        _inc(C, 'order_sell_ok')
        return True
    except Exception as e:
        _log_error(C, f"{stock} 卖出下单失败: {e}")
        _inc(C, 'order_sell_fail')
        return False


def do_buy(C, stock, price, amount, time_str, reason):
    st = _ensure_pos(C, stock)
    vol = _calc_vol(amount, price)
    if vol < 100:
        _log(C, LOG_WARN, 'ORDER', f"{stock} 买入股数不足1手 price={price:.3f} amount={amount} → vol={vol}")
        return False
    if _already_signaled(C, stock, time_str, 'BUY'):
        return False
    if not _passorder_buy(C, stock, vol):
        return False
    # 乐观更新自管状态（实际成交以券商为准，次日可对账）
    old_vol = st['vol']
    old_cost = st['cost']
    if old_vol <= 0:
        st['cost'] = price
        st['vol'] = vol
    else:
        total = old_vol + vol
        st['cost'] = (old_cost * old_vol + price * vol) / total
        st['vol'] = total
    st['high_since_entry'] = max(float(st.get('high_since_entry') or 0), price)
    st['target_pos'] = 1
    st['bars_held'] = 0
    _log(C, LOG_INFO, 'SIGNAL',
         f"{stock} {time_str} 【开多】价={price:.3f} 量={vol} 原因={reason} | {_pos_snap(st, price)}")
    _inc(C, 'signal_buy')
    return True


def do_sell_all(C, stock, price, time_str, reason):
    st = _ensure_pos(C, stock)
    vol = int(st.get('vol') or 0)
    if vol <= 0:
        # 尝试用券商持仓
        vol = _get_broker_position(C, stock)
        if vol <= 0:
            _log(C, LOG_DEBUG, 'ORDER', f"{stock} 无持仓可卖")
            st['target_pos'] = 0
            return False
    vol = _lot_round(vol)
    if vol < 100:
        _log(C, LOG_WARN, 'ORDER', f"{stock} 可卖股数不足1手 vol={vol}")
        return False
    if _already_signaled(C, stock, time_str, 'SELL'):
        return False
    if not _passorder_sell(C, stock, vol):
        return False
    st['vol'] = 0
    st['cost'] = 0.0
    st['target_pos'] = 0
    st['high_since_entry'] = 0.0
    st['bars_held'] = 0
    _log(C, LOG_INFO, 'SIGNAL',
         f"{stock} {time_str} 【平仓】价={price:.3f} 量={vol} 原因={reason} | {_pos_snap(st, price)}")
    _inc(C, 'signal_sell')
    return True


# ---------------------------------------------------------------------------
# 取数辅助（兼容多种返回格式）
# ---------------------------------------------------------------------------

def _get_field_series(C, stock, field, length):
    """获取单个字段历史序列，返回 np.array 或 None。"""
    try:
        data = C.get_history_data(length, '1d', field)
        if data is None:
            return None
        if stock in data:
            arr = np.array(data[stock], dtype=float)
            return arr
        # 有些环境返回 list 按股票池顺序
        if isinstance(data, (list, tuple)) and len(data) > 0:
            return np.array(data, dtype=float)
    except Exception as e:
        _log(C, LOG_DEBUG, 'SKIP', f"{stock} get_history_data({field}) 异常: {e}")
    return None


# ---------------------------------------------------------------------------
# init / handlebar
# ---------------------------------------------------------------------------

def init(C):
    print("=" * 60)
    print("RSRS 统一策略（开关控制版）启动 — 国金QMT 日线")
    print("基于 RSRS_SI.md | 只做多 | 收盘算信号 → 次日执行")
    print("=" * 60)

    # ========== 1. 预设选择 ==========
    # 'A' = 修正标准分（跑通）  'B' = 成交额加权钝化（正式推荐）
    C.preset = 'A'

    # ========== 2. 核心参数（可被预设覆盖）==========
    C.N = 18          # 回归窗口
    C.M = 600         # 标准分 / 波动分位窗口（约3年）
    C.S = 0.9         # 开平阈值（主题偏钝；宽基常用0.7）
    C.tier = 'revise' # slope | zscore | revise | positive
    C.use_amount_weight = False
    C.use_dampen = False
    C.use_ma_filter = False
    C.use_vol_corr_filter = False

    # 应用预设
    if C.preset == 'A':
        C.tier = 'revise'
        C.use_amount_weight = False
        C.use_dampen = False
        C.S = 0.7
        print("[预设A] 修正标准分：tier=revise, 无加权无钝化, S=0.9")
    elif C.preset == 'B':
        C.tier = 'revise'
        C.use_amount_weight = True
        C.use_dampen = True
        C.S = 0.9
        print("[预设B] 成交额加权钝化：tier=revise + amount_w + dampen, S=0.9")
    else:
        print(f"[自定义] tier={C.tier} amount_w={C.use_amount_weight} dampen={C.use_dampen} S={C.S}")

    # ========== 3. 交易与资金 ==========
    C.enable_order = False          # True=真实/模拟下单；False=只出信号
    C.account = '你的资金账号'       # 实盘/模拟请填写
    C.amount_per_etf = 100000        # 每个赛道开仓目标金额（元）
    # 也可按总资金比例，这里简化为固定金额

    # ========== 4. 标的池（主题 ETF，可自行增删）==========
    # 注意：使用 ETF 自身 high/low/amount，不要用宽基信号替行业仓
    C.stock_list = [
        '603311.SH',   # 金海高科
        #'512480.SH',   # 半导体ETF
        #'159819.SZ',   # 人工智能ETF
        #'562500.SH',   # 机器人
        #'159992.SZ',   # 医药ETF
        #'512660.SH',   # 军工ETF
    ]
    # 上市较短的标的会自动因 M 不足而信号为 NaN（保持空仓），可单独降低 M 或提高 S

    try:
        C.set_universe(C.stock_list)
        print(f"已设置股票池: {C.stock_list}")
    except Exception:
        print("set_universe 不支持，将直接按列表取数")

    # ========== 5. 历史长度 ==========
    # 需要 M + N + 缓冲；若回测起点历史不足，信号会 NaN
    C.hist_len = max(C.M + C.N + 50, 700)

    # ========== 6. 日志开关 ==========
    C.log_enable = True
    C.log_level = 3          # 正式运行用 3；排查用 4 或 5
    C.log_to_console = True
    C.log_to_file = False
    C.log_file_path = r'D:\projects\qmt\rsrs.log'  # 例如 r'D:\qmt_log\rsrs.log'
    C.log_bar_every_n = 1
    C.log_mod_bar = False
    C.log_mod_pos = True
    C.log_mod_signal = True
    C.log_mod_decision = True
    C.log_mod_order = True
    C.log_mod_skip = True
    C.log_mod_stat = False

    # ========== 7. 运行时状态 ==========
    C.pos_state = {}
    C.signal_seen = set()
    C.stats = {}
    C._bar_callback_count = 0
    C._last_stat_day = ''

    # 打包 config 字典，便于函数传递
    C.cfg = {
        'N': C.N,
        'M': C.M,
        'S': C.S,
        'tier': C.tier,
        'use_amount_weight': C.use_amount_weight,
        'use_dampen': C.use_dampen,
        'use_ma_filter': C.use_ma_filter,
        'use_vol_corr_filter': C.use_vol_corr_filter,
    }
    #load_preset(C,'A')

    print(f"参数: N={C.N} M={C.M} S={C.S} tier={C.tier}")
    print(f"加强: amount_weight={C.use_amount_weight} dampen={C.use_dampen} "
          f"ma_filter={C.use_ma_filter} vol_corr_filter={C.use_vol_corr_filter}")
    print(f"每标的金额={C.amount_per_etf}  enable_order={C.enable_order}")
    print(f"历史长度={C.hist_len}  股票池数量={len(C.stock_list)}")
    print("策略初始化完成，等待 handlebar ...")
    print("-" * 60)


def handlebar(C):
    C._bar_callback_count = int(getattr(C, '_bar_callback_count', 0)) + 1
    dt, time_str = _parse_bar_time(C)
    is_unclosed = _is_realtime_unclosed_bar(C)
    sig_idx, prev_idx = _signal_index(C)

    # 每日心跳统计
    if time_str != getattr(C, '_last_stat_day', ''):
        C._last_stat_day = time_str
        if C.stats:
            _log(C, LOG_INFO, 'STAT',
                 f"日切 {time_str} | stats={dict(C.stats)}")

    for stock in C.stock_list:
        try:
            st = _ensure_pos(C, stock)

            # ---- 取数 ----
            high = _get_field_series(C, stock, 'high', C.hist_len)
            low = _get_field_series(C, stock, 'low', C.hist_len)
            close = _get_field_series(C, stock, 'close', C.hist_len)
            amount = None
            if C.cfg['use_amount_weight']:
                amount = _get_field_series(C, stock, 'amount', C.hist_len)
                if amount is None:
                    # 尝试 volume * close 近似
                    vol = _get_field_series(C, stock, 'volume', C.hist_len)
                    if vol is not None and close is not None and len(vol) == len(close):
                        amount = vol * close
                        _log(C, LOG_DEBUG, 'SKIP', f"{stock} 无amount，用 volume*close 近似")

            if high is None or low is None or close is None:
                _log(C, LOG_DETAIL, 'SKIP', f"{stock} {time_str} 关键字段缺失，跳过")
                _inc(C, 'skip_data')
                continue

            # 对齐长度（取最短）
            min_len = min(len(high), len(low), len(close))
            if amount is not None:
                min_len = min(min_len, len(amount))
            if min_len < C.N + 10:
                _log(C, LOG_DETAIL, 'SKIP',
                     f"{stock} {time_str} 数据过短 len={min_len} < N+10，跳过")
                _inc(C, 'skip_short')
                continue

            high = high[-min_len:]
            low = low[-min_len:]
            close = close[-min_len:]
            if amount is not None:
                amount = amount[-min_len:]

            # 决策索引（相对当前数组）
            t = len(close) + sig_idx   # sig_idx 为 -1 或 -2
            if t < C.N - 1:
                _log(C, LOG_DETAIL, 'SKIP', f"{stock} {time_str} 决策点历史不足 t={t}")
                continue

            price = float(close[t])
            if price <= 0 or np.isnan(price):
                continue

            # ---- 计算信号 ----
            x, beta, r2, z, q = compute_rsrs_signal(
                C.cfg, high, low, close, amount, t
            )

            # 可选 MA 过滤用数据
            ma20 = ma20_prev3 = None
            if C.cfg['use_ma_filter'] and t >= 20:
                ma20 = np.mean(close[t - 19: t + 1])
                if t >= 23:
                    ma20_prev3 = np.mean(close[t - 22: t - 2])

            # 可选量价相关（简化：近10日 volume 与 近10日 x 的相关，这里用 close 变化近似）
            vol_corr = None
            # 若需要可扩展，当前默认关

            old_pos = int(st.get('target_pos', 0))
            new_pos = next_position(
                old_pos, x, C.cfg['S'], C.cfg,
                ma20=ma20, ma20_prev3=ma20_prev3, vol_corr=vol_corr
            )

            # 日志（DETAIL 级）
            q_s = f"{q:.3f}" if not (isinstance(q, float) and np.isnan(q)) else "N/A"
            z_s = f"{z:.3f}" if not (isinstance(z, float) and np.isnan(z)) else "N/A"
            x_s = f"{x:.4f}" if not (isinstance(x, float) and np.isnan(x)) else "nan"
            beta_s = f"{beta:.4f}" if not (isinstance(beta, float) and np.isnan(beta)) else "nan"
            r2_s = f"{r2:.3f}" if not (isinstance(r2, float) and np.isnan(r2)) else "nan"
            _log(C, LOG_DETAIL, 'BAR',
                 f"{stock} {time_str} px={price:.3f} x={x_s} "
                 f"β={beta_s} R²={r2_s} "
                 f"z={z_s} q={q_s} pos={old_pos}→{new_pos} unclosed={is_unclosed}")

            # 持有天数累计
            bar_key = time_str
            if st.get('last_bar_key') != bar_key and old_pos == 1:
                st['bars_held'] = int(st.get('bars_held', 0)) + 1
                st['last_bar_key'] = bar_key
                if price > float(st.get('high_since_entry') or 0):
                    st['high_since_entry'] = price

            # ---- 状态机驱动交易 ----
            if new_pos != old_pos:
                if new_pos == 1 and old_pos == 0:
                    # 开多
                    reason = (f"x={x:.3f}>{C.cfg['S']} tier={C.cfg['tier']} "
                              f"β={beta:.3f} z={z_s} R²={r2:.3f}")
                    do_buy(C, stock, price, C.amount_per_etf, time_str, reason)
                elif new_pos == 0 and old_pos == 1:
                    # 平仓
                    reason = (f"x={x:.3f}<{-C.cfg['S']} tier={C.cfg['tier']} "
                              f"β={beta:.3f} z={z_s}")
                    do_sell_all(C, stock, price, time_str, reason)
                # 更新目标（即使下单失败也记录意图，便于对账）
                st['target_pos'] = new_pos
            else:
                # 中间持有 or 继续空仓
                if old_pos == 1 and not (isinstance(x, float) and np.isnan(x)):
                    _log(C, LOG_DEBUG, 'DECISION',
                         f"{stock} {time_str} 持有中 x={x:.3f} 在 [-S,+S] 内，继续持有")

            # 对账提示（券商持仓 vs 自管）
            if C.log_level >= LOG_DETAIL:
                broker_vol = _get_broker_position(C, stock)
                if broker_vol != st.get('vol', 0):
                    _log(C, LOG_DEBUG, 'POS',
                         f"{stock} 对账差异: 自管vol={st.get('vol')} 券商vol={broker_vol}")

        except Exception as e:
            _log_error(C, f"{stock} handlebar 异常: {type(e).__name__}: {e}")
            import traceback
            _log(C, LOG_DEBUG, 'ERROR', traceback.format_exc())

    # 可选：全局统计输出
    if C._bar_callback_count % 20 == 0 and C.stats:
        _log(C, LOG_DETAIL, 'STAT', f"累计回调={C._bar_callback_count} stats={dict(C.stats)}")


# =============================================================================
# 快速切换预设示例（可在 init 外或回测脚本中调用）
# =============================================================================
def load_preset(C, name):
    """运行时切换预设（回测研究用）。实盘请冻结一组。"""
    name = name.upper()
    if name == 'A':
        C.cfg['tier'] = 'revise'
        C.cfg['use_amount_weight'] = False
        C.cfg['use_dampen'] = False
        C.cfg['S'] = 0.9
        C.tier = 'revise'
        C.use_amount_weight = False
        C.use_dampen = False
        C.S = 0.9
        print("[load_preset] A 修正标准分")
    elif name == 'B':
        C.cfg['tier'] = 'revise'
        C.cfg['use_amount_weight'] = True
        C.cfg['use_dampen'] = True
        C.cfg['S'] = 0.9
        C.tier = 'revise'
        C.use_amount_weight = True
        C.use_dampen = True
        C.S = 0.9
        print("[load_preset] B 成交额加权钝化")
    elif name == 'C':
        C.cfg['tier'] = 'zscore'
        C.cfg['use_amount_weight'] = False
        C.cfg['use_dampen'] = False
        C.cfg['S'] = 0.8
        print("[load_preset] C 标准分基线")
    elif name == 'D':
        C.cfg['tier'] = 'positive'
        C.cfg['use_amount_weight'] = False
        C.cfg['use_dampen'] = False
        C.cfg['S'] = 0.8
        print("[load_preset] D 右偏研究")
    else:
        print(f"未知预设 {name}，保持当前")


# =============================================================================
# 结束
# =============================================================================
