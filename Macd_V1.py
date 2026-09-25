# -*- coding: gbk -*-
# =============================================================================
# 日线 MACD趋势过滤 + KDJ/RSI市场状态自适应超卖反转策略（开关控制版）
# 适用：国金证券 QMT / 迅投 Python 模型
# 请确认模型周期为【日线 / 1d】
# =============================================================================
#
# 【一句话】
#   MACD绿色柱（负值）缩短做趋势过滤；根据市场状态（ADX）自动切换 KDJ（震荡）或 RSI（趋势）
#   超卖反转买入；卖出优先止损/止盈（均可开关）/移动止盈 → MACD顶背离全平 → 死叉半仓。
#
# 【核心逻辑】
#   买：MACD连续为负且当前绿色柱比前一根缩短；KDJ/RSI触发后可按次数上限加仓
#       + 震荡市：KDJ金叉 且 (K<30 或 J<20)
#       + 趋势市：RSI≤30 且拐头向上
#       + both模式：以上任一满足
#       + 可选成交量放量过滤（默认关）
#   卖优先级：止损 > 止盈 > 移动止盈 > MACD顶背离(100%) > MACD死叉(���配置比例) > 时间止损
#
# 【使用说明】
#   1. 回测：TRADE_MODE = 'backtest'，向QMT回测引擎提交模拟委托
#   2. 人工确认：TRADE_MODE = 'notify'，发送企业微信通知但不委托
#   3. 自动委托：TRADE_MODE = 'auto'，盘后生成信号，次交易日定时委托
# =============================================================================

import json
import os
from pathlib import Path
import numpy as np # type: ignore
import talib # type: ignore
from datetime import datetime, timedelta


# =============================================================================
# ========================【参数配置区】直接在这里修改========================
# =============================================================================

# ----- 账户与下单 -----
TRADE_MODE       = 'auto'          # 'backtest' / 'notify' / 'auto'
LOCAL_CONFIG_PATH = Path(r'c:\users\administrator\downloads\qmt.local.json')

# ----- 标的池 -----
STOCK_LIST = [
    "603311.SH",  # 金海高科
    # '512480.SH',   # 半导体ETF 国联安
    # '159819.SZ',   # 人工智能ETF 易方达
    # '562500.SH',   # 机器人ETF 华夏
    # '159992.SZ',   # 创新药ETF 银华
    # '512660.SH',   # 军工ETF 国泰
]

# ----- 买入金额 -----
BUY_AMOUNT       = 10000           # 单标的单次买入金额（元）
ENABLE_REPEAT_BUY = True           # True=允许持仓期间重复买入
MAX_BUY_COUNT     = 2              # 单个完整持仓周期最多买入次数

# ----- MACD -----
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9

# ----- KDJ -----
KDJ_N            = 9
KDJ_M1           = 3
KDJ_M2           = 3
KDJ_K_OVERSOLD   = 30              # K < 此值视为超卖
KDJ_J_OVERSOLD   = 20              # J < 此值视为超卖

# ----- RSI -----
RSI_PERIOD       = 6               # 与历史策略一致，可改14
RSI_OVERSOLD     = 35

# ----- 市场状态（ADX） -----
ADX_PERIOD       = 14
ADX_TREND        = 25              # >25 判定为趋势市
ADX_RANGE        = 20              # <20 判定为震荡市
REGIME_MODE      = 'auto'          # 'auto' / 'kdj' / 'rsi' / 'both'
                                   # auto: 根据ADX自动切换
                                   # kdj : 强制只用KDJ
                                   # rsi : 强制只用RSI
                                   # both: KDJ或RSI任一满足即买

# ----- 成交量过滤 -----
USE_VOLUME_FILTER = False          # 默认关闭
VOLUME_MULT       = 1.2            # 放量倍数（当日成交额 >= 均额 * 此值）
VOLUME_MA_PERIOD  = 20

# ----- 止损止盈 -----
USE_STOP_LOSS    = False           # 是否启用止损
USE_TAKE_PROFIT  = False           # 是否启用止盈
STOP_MODE         = 'pct'          # 'pct' 固定百分比  或  'atr' 动态ATR
STOP_LOSS_PCT     = 0.08           # 固定止损 8%
TAKE_PROFIT_PCT   = 0.18           # 固定止盈 18%
ATR_PERIOD        = 14
STOP_ATR_MULT     = 2.0            # ATR止损倍数
TAKE_ATR_MULT     = 3.5            # ATR止盈倍数

# ----- 移动止盈 -----
USE_TRAILING      = False          # 是否启用移动止盈
TRAIL_PCT         = 0.08           # 从最高点回撤 8%

# ----- 死叉卖出比例 -----
DEATH_CROSS_SELL_RATIO = 0.5       # 0.3~1.0，死叉时卖出比例

# ----- 时间止损 -----
USE_TIME_STOP     = False
MAX_HOLD_DAYS     = 30

# ----- 顶背离参数 -----
DIV_LOOKBACK      = 30
DIV_PEAK_ORDER    = 3

# ----- 日志 -----
LOG_ENABLE        = True
LOG_LEVEL         = 3              # 0=OFF 1=ERROR 2=WARN 3=INFO 4=DETAIL 5=DEBUG
LOG_TO_CONSOLE    = True
LOG_TO_FILE       = True
LOG_FILE_PATH     = r'C:\qmt_log\macd_kdj_rsi.log'
LOG_BAR_EVERY_N   = 1              # DETAIL 日志已按交易日去重；必须保持 1
GITHUB_LOG_REPOSITORY = 'ericqnli/qmt-signals'
GITHUB_LOG_DIRECTORY  = 'daily'

# ----- 企业微信日线状态推送 -----
# 仅在盘后（15:05 起）推送一次。
POST_MARKET_NOTIFY_START = (15, 5)
SEND_SIGNAL_NOTIFICATIONS = False  # 买卖信号仅写入盘后汇总，不单独即时推送

# ----- 自动委托时点 -----
# auto 模式：盘后按完整日线记录信号；下一交易日到达该时间后按最新价提交。
AUTO_EXECUTION_TIME = (9, 35)
PENDING_ORDER_FILE_PATH = r'C:\qmt_log\macd_pending_orders.json'

LOG_MOD_BAR       = True
LOG_MOD_POS       = True
LOG_MOD_SIGNAL    = True
LOG_MOD_DECISION  = True
LOG_MOD_ORDER     = True
LOG_MOD_SKIP      = True
LOG_MOD_STAT      = True

# =============================================================================
# ========================【参数配置区结束】===================================
# =============================================================================


def _load_local_config():
    """加载本机账户、企业微信和 GitHub 配置；该文件必须保持在 Git 忽略列表中。"""
    try:
        with LOCAL_CONFIG_PATH.open('r', encoding='utf-8') as file:
            config = json.load(file)
    except FileNotFoundError as error:
        raise RuntimeError(f"未找到本地配置文件: {LOCAL_CONFIG_PATH}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"本地配置文件不是有效 JSON: {LOCAL_CONFIG_PATH}") from error
    except OSError as error:
        raise RuntimeError(f"无法读取本地配置文件: {LOCAL_CONFIG_PATH}") from error

    if not isinstance(config, dict):
        raise ValueError("qmt.local.json 顶层必须是 JSON 对象")

    account_id = str(config.get('account_id', '')).strip()
    webhook_url = str(config.get('wecom_webhook_url', '')).strip()
    github_token = str(config.get('github_token', '')).strip()
    if not account_id:
        raise ValueError("qmt.local.json 缺少 account_id")
    if not webhook_url:
        raise ValueError("qmt.local.json 缺少 wecom_webhook_url")
    if not github_token:
        raise ValueError("qmt.local.json 缺少 github_token")
    return account_id, webhook_url, github_token


# ---------------------------------------------------------------------------
# 日志系统
# ---------------------------------------------------------------------------
LOG_OFF, LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DETAIL, LOG_DEBUG = 0, 1, 2, 3, 4, 5
_LEVEL_NAME = {0: 'OFF', 1: 'ERROR', 2: 'WARN', 3: 'INFO', 4: 'DETAIL', 5: 'DEBUG'}
_detail_bar_seen = set()
_status_notify_seen = set()


def _daily_status_flag_path(time_str, period):
    base = os.path.dirname(LOG_FILE_PATH or '') or '.'
    return os.path.join(base, f'.wechat_status_{period}_{time_str}')


def _already_sent_daily_status(time_str, period):
    if not time_str or time_str == '----':
        return True
    key = (period, time_str)
    if key in _status_notify_seen:
        return True
    path = _daily_status_flag_path(time_str, period)
    if os.path.isfile(path):
        _status_notify_seen.add(key)
        return True
    return False


def _mark_daily_status_sent(time_str, period):
    _status_notify_seen.add((period, time_str))
    try:
        path = _daily_status_flag_path(time_str, period)
        folder = os.path.dirname(path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(datetime.now().isoformat())
    except Exception as error:
        _log_warn(None, f"{period} {time_str} 日线状态标记写入失败: {error}")


def _status_notification_period(now):
    """盘后才允许推送日线状态；其他时间不发送。"""
    clock = (now.hour, now.minute)
    if clock >= POST_MARKET_NOTIFY_START:
        return '盘后'
    return None


def _clock_reached(now, target):
    return (now.hour, now.minute) >= target


def _validate_clock(value, name):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} 必须是 (时, 分)，当前为 {value!r}")
    hour, minute = value
    if (not isinstance(hour, int) or not isinstance(minute, int)
            or not 0 <= hour <= 23 or not 0 <= minute <= 59):
        raise ValueError(f"{name} 必须是有效的 (时, 分)，当前为 {value!r}")
    return hour, minute


def _qmt_time(clock):
    hour, minute = clock
    return f'{hour:02d}{minute:02d}00'


def _qmt_datetime(clock, date_str=None):
    """生成 QMT 可接受的运行时间字符串：YYYY-MM-DD HH:MM:SS。"""
    hour, minute = _validate_clock(clock, 'clock')
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    return f'{date_str} {hour:02d}:{minute:02d}:00'


def _load_pending_orders(C):
    path = C.pending_order_file_path
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as file:
            orders = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取待执行委托文件: {path}") from error
    if not isinstance(orders, list) or not all(isinstance(order, dict) for order in orders):
        raise ValueError(f"待执行委托文件格式错误: {path}")
    return orders


def _save_pending_orders(C):
    path = C.pending_order_file_path
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    temp_path = f'{path}.tmp'
    try:
        with open(temp_path, 'w', encoding='utf-8') as file:
            json.dump(C.pending_orders, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    except OSError as error:
        raise RuntimeError(f"无法保存待执行委托文件: {path}") from error


def _queue_pending_order(C, order):
    key = (order['signal_date'], order['stock'], order['side'])
    for pending in C.pending_orders:
        if (pending.get('signal_date'), pending.get('stock'), pending.get('side')) == key:
            _log(C, LOG_DEBUG, 'SIGNAL',
                 f"{order['stock']} {order['signal_date']} {order['side']} 已在待执行队列")
            return False
    C.pending_orders.append(order)
    try:
        _save_pending_orders(C)
    except Exception:
        C.pending_orders.pop()
        raise
    return True


def _submit_due_pending_orders(C, now):
    if (C.trade_mode != 'auto'
            or not _clock_reached(now, C.auto_execution_time)
            or _clock_reached(now, C.post_market_notify_start)):
        return

    today = now.strftime('%Y-%m-%d')
    remaining = []
    changed = False
    for order in C.pending_orders:
        signal_date = str(order.get('signal_date', ''))
        if not signal_date or signal_date >= today:
            remaining.append(order)
            continue

        stock = str(order.get('stock', ''))
        side = str(order.get('side', ''))
        try:
            volume = int(order.get('volume', 0))
        except (TypeError, ValueError):
            volume = 0
        if not stock or side not in ('BUY', 'SELL') or volume < 100:
            _log_error(C, f"待执行委托格式错误，已丢弃: {order!r}")
            changed = True
            continue

        if side == 'SELL':
            try:
                broker_pos = _get_broker_position_volume(C, stock)
            except Exception as error:
                _log_error(C, f"{stock} 查询卖出持仓失败，已取消待执行委托: {error}")
                changed = True
                continue
            volume = min(volume, (broker_pos // 100) * 100)
            if volume < 100:
                _log_warn(C, f"{stock} 可卖持仓不足，已取消待执行卖出委托")
                changed = True
                continue

        try:
            passorder(23 if side == 'BUY' else 24, 1101, C.account, stock, 14, -1, volume, # type: ignore
                      'MACD_KDJ_RSI', 1, '', C)
        except Exception as error:
            _log_error(C, f"{stock} 待执行{side}委托失败，已取消待执行委托: {error}")
            changed = True
            continue

        changed = True
        _log(C, LOG_INFO, 'ORDER',
             f"{stock} 待执行{side}委托已提交 vol={volume} 信号日={signal_date}；"
             "等待券商成交回报后同步持仓")
        _inc(C, f'order_{side.lower()}')

    if changed:
        C.pending_orders = remaining
        _save_pending_orders(C)


def _get_weekly_log_path(base_path):
    """按周滚动日志文件路径。"""
    if not base_path:
        return ''
    try:
        now = datetime.now()
        monday = now - timedelta(days=now.weekday())
        week_tag = monday.strftime('%Y%m%d')
        dir_name = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        stem, ext = os.path.splitext(base_name)
        if not ext:
            ext = '.log'
        weekly_name = f"{stem}_{week_tag}{ext}"
        full = os.path.join(dir_name, weekly_name) if dir_name else weekly_name
        if dir_name and not os.path.isdir(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        return full
    except Exception:
        return base_path


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
        path = _get_weekly_log_path(getattr(C, 'log_file_path', '') or '')
        if path:
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(line + '\n')
            except Exception as e:
                if not getattr(C, '_log_file_error_reported', False):
                    C._log_file_error_reported = True
                    print(f"[日志] 写文件失败: {e} path={path}")


def _log_error(C, msg):
    _log(C, LOG_ERROR, 'ERROR', msg)


def _log_warn(C, msg):
    _log(C, LOG_WARN, 'WARN', msg)


def _should_log_detail_bar(stock, time_str):
    """同一标的在同一交易日只输出一次 DETAIL 指标快照。"""
    key = (stock, time_str)
    if key in _detail_bar_seen:
        return False
    _detail_bar_seen.add(key)
    if len(_detail_bar_seen) > 5000:
        _detail_bar_seen.clear()
        _detail_bar_seen.add(key)
    return True


def _inc(C, key, n=1):
    if not hasattr(C, 'stats') or C.stats is None:
        C.stats = {}
    C.stats[key] = int(C.stats.get(key, 0)) + n


def _send_wechat_notification(C, title, content):
    """通过企业微信机器人发送人工确认通知。"""
    webhook_url = getattr(C, 'wechat_work_webhook_url', '')
    if not webhook_url:
        _log_warn(C, f"{title} 未发送：WECHAT_WORK_WEBHOOK_URL 未配置")
        return False
    try:
        import json
        from urllib.request import Request, urlopen

        payload = json.dumps(
            {'msgtype': 'text', 'text': {'content': f'{title}\n{content}'}},
            ensure_ascii=False,
        ).encode('utf-8')
        request = Request(
            webhook_url,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status}')
            result = json.loads(response.read().decode('utf-8'))
        if not isinstance(result, dict) or result.get('errcode') != 0:
            raise RuntimeError(
                f"企业微信返回错误: {result.get('errmsg', result) if isinstance(result, dict) else result}"
            )
        _log(C, LOG_INFO, 'SIGNAL', f"{title} 企业微信通知已发送")
        _inc(C, 'wechat_notify_ok')
        return True
    except Exception as e:
        _log_error(C, f"{title} 企业微信通知失败: {type(e).__name__}: {e}")
        _inc(C, 'wechat_notify_fail')
        return False


def _upload_log_to_github(C):
    """将当前周日志覆盖上传到 GitHub，保留每个交易日收盘后的最新快照。"""
    if not getattr(C, 'log_to_file', False):
        _log_warn(C, "GitHub日志未上传：LOG_TO_FILE=False")
        return False

    log_path = _get_weekly_log_path(getattr(C, 'log_file_path', '') or '')
    if not log_path or not os.path.isfile(log_path):
        _log_warn(C, f"GitHub日志未上传：日志文件不存在 path={log_path}")
        return False

    repository = getattr(C, 'github_log_repository', '')
    token = getattr(C, 'github_token', '')
    if not repository or not token:
        _log_warn(C, "GitHub日志未上传：仓库或令牌未配置")
        return False

    try:
        import base64
        from urllib.error import HTTPError
        from urllib.parse import quote
        from urllib.request import Request, urlopen

        remote_path = '/'.join([
            quote(getattr(C, 'github_log_directory', 'daily').strip('/'), safe=''),
            quote(os.path.basename(log_path), safe=''),
        ])
        api_url = f'https://api.github.com/repos/{repository}/contents/{remote_path}'
        headers = {
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {token}',
            'X-GitHub-Api-Version': '2022-11-28',
        }

        sha = None
        try:
            with urlopen(Request(api_url, headers=headers), timeout=15) as response:
                existing = json.loads(response.read().decode('utf-8'))
                sha = existing.get('sha')
        except HTTPError as error:
            if error.code != 404:
                raise

        with open(log_path, 'rb') as file:
            encoded_content = base64.b64encode(file.read()).decode('ascii')
        payload = {
            'message': f'Update QMT log: {os.path.basename(log_path)}',
            'content': encoded_content,
        }
        if sha:
            payload['sha'] = sha
        request = Request(
            api_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={**headers, 'Content-Type': 'application/json'},
            method='PUT',
        )
        with urlopen(request, timeout=30) as response:
            if response.status not in (200, 201):
                raise RuntimeError(f'HTTP {response.status}')
        _log(C, LOG_INFO, 'SYS', f"GitHub日志已上传: {repository}/{remote_path}")
        _inc(C, 'github_log_upload_ok')
        return True
    except Exception as error:
        _log_error(C, f"GitHub日志上传失败: {type(error).__name__}: {error}")
        _inc(C, 'github_log_upload_fail')
        return False


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _parse_bar_time(C, offset=0):
    try:
        bar_time = C.get_bar_timetag(C.barpos + offset)
        if bar_time > 1e12:
            dt = datetime.fromtimestamp(bar_time / 1000.0)
        else:
            dt = datetime.fromtimestamp(bar_time)
        return dt, dt.strftime('%Y-%m-%d')
    except Exception:
        return None, "----"


def _is_realtime_unclosed_bar(C):
    try:
        if hasattr(C, 'is_last_bar') and callable(C.is_last_bar):
            return bool(C.is_last_bar())
        if hasattr(C, 'is_last_bar'):
            return bool(C.is_last_bar)
    except Exception:
        pass
    return False


def _signal_index(C, now=None):
    """返回用于决策的已收盘索引。实盘最新未收盘用 -2。"""
    now = now or datetime.now()
    if _is_realtime_unclosed_bar(C) and (now.hour, now.minute) < POST_MARKET_NOTIFY_START:
        return -2, -3
    return -1, -2


def _calc_volume_by_amount(amount, price):
    """按金额向下取整到100份。"""
    if price is None or price <= 0 or amount <= 0:
        return 0
    return int(amount / price / 100) * 100


def _position_volume(position):
    """兼容 QMT 返回的数值、对象和字典持仓结构。"""
    if position is None:
        return 0
    if isinstance(position, (int, float, np.integer, np.floating)):
        return max(0, int(position))
    names = ('can_use_volume', 'volume', 'available_volume', 'total_volume')
    if isinstance(position, dict):
        for name in names:
            if name in position and position[name] is not None:
                return max(0, int(float(position[name])))
    else:
        for name in names:
            value = getattr(position, name, None)
            if value is not None:
                return max(0, int(float(value)))
    raise ValueError(f"无法识别的持仓数据: {position!r}")


def _get_broker_position_volume(C, stock):
    if not hasattr(C, 'get_position'):
        return 0
    return _position_volume(C.get_position(stock))


def _reset_position_state(st):
    st['vol'] = 0
    st['buy_price'] = 0.0
    st['buy_time'] = ''
    st['bars_held'] = 0
    st['high_since_entry'] = 0.0
    st['entry_atr'] = 0.0
    st['buy_count'] = 0


def _already_signaled(C, stock, time_str, side):
    key = (stock, time_str, side)
    if key in C.signal_seen:
        _log(C, LOG_DEBUG, 'SIGNAL', f"{stock} {time_str} {side} 已触发过，去重跳过")
        _inc(C, 'signal_dedup')
        return True
    C.signal_seen.add(key)
    if len(C.signal_seen) > 5000:
        C.signal_seen = set(list(C.signal_seen)[-2000:])
    return False


def _ensure_pos_state(C, stock):
    if stock not in C.pos_state:
        C.pos_state[stock] = {
            'vol': 0,
            'buy_price': 0.0,
            'buy_time': '',
            'bars_held': 0,
            'high_since_entry': 0.0,
            'entry_atr': 0.0,
            'buy_count': 0,
        }
    return C.pos_state[stock]


def _pnl_pct(buy_price, curr_price):
    if buy_price is None or buy_price <= 0 or curr_price is None:
        return None
    return (curr_price - buy_price) / buy_price

def _sma_tdx(series, n, m=1):
    """通达信 SMA(X,N,M)=(M*X+(N-M)*Y')/N。首个有效值作为初值。"""
    x = np.asarray(series, dtype=float)
    y = np.full(x.shape, np.nan, dtype=float)
    prev = None
    n = float(n)
    m = float(m)
    for i, v in enumerate(x):
        if np.isnan(v):
            if prev is not None:
                y[i] = prev
            continue
        if prev is None:
            prev = v
        else:
            prev = (m * v + (n - m) * prev) / n
        y[i] = prev
    return y


def _kdj_tdx(high, low, close, n=9, m1=3, m2=3):
    """通达信 KDJ(N,M1,M2)。返回 K, D, J。"""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    llv = talib.MIN(low, timeperiod=n)
    hhv = talib.MAX(high, timeperiod=n)
    denom = hhv - llv
    rsv = np.where(denom > 0, (close - llv) / denom * 100.0, 50.0)
    rsv = np.where(np.isnan(llv) | np.isnan(hhv), np.nan, rsv)
    k = _sma_tdx(rsv, m1, 1)
    d = _sma_tdx(k, m2, 1)
    j = 3.0 * k - 2.0 * d
    return k, d, j

def _detect_macd_top_divergence(close, dif, lookback=30, peak_order=3):
    """简单顶背离检测：价格创新高但DIF高点降低。"""
    if len(close) < lookback + 5 or len(dif) < lookback + 5:
        return False
    c = np.asarray(close[-lookback:], dtype=float)
    d = np.asarray(dif[-lookback:], dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isnan(d)):
        return False

    peaks_idx = []
    for i in range(peak_order, len(c) - peak_order):
        if c[i] == np.max(c[i - peak_order:i + peak_order + 1]):
            peaks_idx.append(i)
    if len(peaks_idx) < 2:
        return False
    p1, p2 = peaks_idx[-2], peaks_idx[-1]
    if c[p2] > c[p1] * 1.001 and d[p2] < d[p1] * 0.995:
        return True
    return False


def _get_market_data_safe(C, stock, fields, count=120, end_time=''):
    """安全获取行情：按回测当前日期截断，优先 get_market_data_ex。"""
    try:
        kwargs = dict(
            period='1d',
            count=count,
            dividend_type='front',
            fill_data=True,
        )
        if end_time:
            kwargs['end_time'] = end_time
        md = C.get_market_data_ex(fields, [stock], **kwargs)
        if md is not None and stock in md:
            df = md[stock]
            result = {}
            for f in fields:
                if hasattr(df, 'columns') and f in df.columns:
                    result[f] = np.array(df[f].values, dtype=float)
                elif isinstance(df, dict) and f in df:
                    val = df[f]
                    result[f] = np.array(val.values if hasattr(val, 'values') else val, dtype=float)
                else:
                    result[f] = None
            if result.get('close') is not None and len(result['close']) >= 50:
                return result
    except Exception:
        pass

    try:
        kwargs = dict(stock_code=[stock], period='1d', count=count)
        if end_time:
            kwargs['end_time'] = end_time
        md = C.get_market_data(fields, **kwargs)
        if md is not None:
            result = {}
            for f in fields:
                if f in md:
                    val = md[f]
                    result[f] = np.array(val.values if hasattr(val, 'values') else val, dtype=float)
                else:
                    result[f] = None
            if result.get('close') is not None and len(result['close']) >= 50:
                return result
    except Exception:
        pass

    try:
        data = C.get_history_data(count, '1d', 'close')
        if stock in data and len(data[stock]) >= 50:
            close = np.array(data[stock], dtype=float)
            result = {'close': close}
            for f in ['high', 'low', 'volume', 'amount']:
                try:
                    d2 = C.get_history_data(count, '1d', f)
                    if stock in d2:
                        result[f] = np.array(d2[stock], dtype=float)
                except Exception:
                    result[f] = None
            return result
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

def init(C):
    _detail_bar_seen.clear()
    print("=" * 60)
    print("日线 MACD趋势 + KDJ/RSI自适应超卖策略 启动")
    print("=" * 60)

    C.account, C.wechat_work_webhook_url, C.github_token = _load_local_config()
    C.set_account(C.account)
    C.trade_mode             = TRADE_MODE.lower()
    if C.trade_mode not in ('backtest', 'notify', 'auto'):
        raise ValueError(
            f"TRADE_MODE 必须是 'backtest'、'notify' 或 'auto'，当前为 {TRADE_MODE!r}"
        )
    C.post_market_notify_start = _validate_clock(
        POST_MARKET_NOTIFY_START, 'POST_MARKET_NOTIFY_START'
    )
    C.auto_execution_time = _validate_clock(AUTO_EXECUTION_TIME, 'AUTO_EXECUTION_TIME')
    if C.auto_execution_time >= C.post_market_notify_start:
        raise ValueError("AUTO_EXECUTION_TIME 必须早于 POST_MARKET_NOTIFY_START")
    C.pending_order_file_path = PENDING_ORDER_FILE_PATH
    C.pending_orders = _load_pending_orders(C)
    C.submit_orders          = C.trade_mode == 'backtest'
    C.stock_list             = STOCK_LIST
    C.buy_amount             = BUY_AMOUNT
    C.enable_repeat_buy      = ENABLE_REPEAT_BUY
    C.max_buy_count          = MAX_BUY_COUNT

    C.macd_fast              = MACD_FAST
    C.macd_slow              = MACD_SLOW
    C.macd_signal            = MACD_SIGNAL

    C.kdj_n                  = KDJ_N
    C.kdj_m1                 = KDJ_M1
    C.kdj_m2                 = KDJ_M2
    C.kdj_k_oversold         = KDJ_K_OVERSOLD
    C.kdj_j_oversold         = KDJ_J_OVERSOLD

    C.rsi_period             = RSI_PERIOD
    C.rsi_oversold           = RSI_OVERSOLD

    C.adx_period             = ADX_PERIOD
    C.adx_trend              = ADX_TREND
    C.adx_range              = ADX_RANGE
    C.regime_mode            = REGIME_MODE

    C.use_volume_filter      = USE_VOLUME_FILTER
    C.volume_mult            = VOLUME_MULT
    C.volume_ma_period       = VOLUME_MA_PERIOD

    C.stop_mode              = STOP_MODE
    C.use_stop_loss          = USE_STOP_LOSS
    C.use_take_profit        = USE_TAKE_PROFIT
    C.stop_loss_pct          = STOP_LOSS_PCT
    C.take_profit_pct        = TAKE_PROFIT_PCT
    C.atr_period             = ATR_PERIOD
    C.stop_atr_mult          = STOP_ATR_MULT
    C.take_atr_mult          = TAKE_ATR_MULT

    C.use_trailing           = USE_TRAILING
    C.trail_pct              = TRAIL_PCT

    C.death_cross_sell_ratio = DEATH_CROSS_SELL_RATIO

    C.use_time_stop          = USE_TIME_STOP
    C.max_hold_days          = MAX_HOLD_DAYS

    C.div_lookback           = DIV_LOOKBACK
    C.div_peak_order         = DIV_PEAK_ORDER

    C.log_enable             = LOG_ENABLE
    C.log_level              = LOG_LEVEL
    C.log_to_console         = LOG_TO_CONSOLE
    C.log_to_file            = LOG_TO_FILE
    C.log_file_path          = LOG_FILE_PATH
    C.log_bar_every_n        = LOG_BAR_EVERY_N
    C.github_log_repository  = GITHUB_LOG_REPOSITORY
    C.github_log_directory   = GITHUB_LOG_DIRECTORY

    C.log_mod_bar            = LOG_MOD_BAR
    C.log_mod_pos            = LOG_MOD_POS
    C.log_mod_signal         = LOG_MOD_SIGNAL
    C.log_mod_decision       = LOG_MOD_DECISION
    C.log_mod_order          = LOG_MOD_ORDER
    C.log_mod_skip           = LOG_MOD_SKIP
    C.log_mod_stat           = LOG_MOD_STAT

    C.pos_state = {}
    C.signal_seen = set()
    C.stats = {}
    C._bar_callback_count = 0
    C._log_file_error_reported = False

    try:
        C.set_universe(C.stock_list)
        print(f"已设置股票池: {C.stock_list}")
    except Exception:
        print("set_universe 不支持，将使用列表循环取数")

    for code in C.stock_list:
        try:
            download_history_data(code, '1d', '20200101', '') # type: ignore
            print(f"已下载历史数据: {code}")
        except Exception as e:
            print(f"下载历史数据失败 {code}: {e}")

    if C.trade_mode in ('notify', 'auto'):
        if not hasattr(C, 'run_time') or not callable(C.run_time):
            raise RuntimeError("QMT Context 未提供 run_time，无法按日线收盘和定时委托运行")
        C.run_time('handlebar', '1nDay', _qmt_datetime(C.post_market_notify_start))
        if C.trade_mode == 'auto':
            C.run_time('handlebar', '1nDay', _qmt_datetime(C.auto_execution_time))

    print(f"regime_mode={C.regime_mode}  stop_mode={C.stop_mode}  buy_amount={C.buy_amount}")
    print(f"use_stop_loss={C.use_stop_loss}  use_take_profit={C.use_take_profit}")
    print(f"enable_repeat_buy={C.enable_repeat_buy}  max_buy_count={C.max_buy_count}")
    print(f"use_volume_filter={C.use_volume_filter}  death_cross_ratio={C.death_cross_sell_ratio}")
    print(f"trade_mode={C.trade_mode}  submit_orders={C.submit_orders}  log_level={C.log_level}")
    if C.trade_mode in ('notify', 'auto') and not C.wechat_work_webhook_url:
        _log_warn(C, "未配置WECHAT_WORK_WEBHOOK_URL，日线状态只会输出到日志")
    if C.trade_mode == 'auto':
        print(
            f"auto模式：盘后{_qmt_time(C.post_market_notify_start)}记录信号并推送，"
            f"次交易日{_qmt_time(C.auto_execution_time)}后提交待执行委托；"
            f"待执行数={len(C.pending_orders)}"
        )
    print("初始化完成，开始运行...")
    print("=" * 60)


def _callback_value(info, *names, default=None):
    if isinstance(info, dict):
        for name in names:
            if name in info:
                return info[name]
    else:
        for name in names:
            value = getattr(info, name, None)
            if value is not None:
                return value
    return default


def order_callback(C, order_info):
    """记录原生 QMT 委托回报；拒单须立即通知人工处理。"""
    status = _callback_value(order_info, 'order_status', 'status')
    stock = _callback_value(order_info, 'stock_code', 'stock', default='未知标的')
    order_id = _callback_value(order_info, 'order_id', 'entrust_no', default='未知编号')
    message = _callback_value(order_info, 'status_msg', 'status_message', 'msg', default='')
    _log(C, LOG_INFO, 'ORDER',
         f"委托回报 stock={stock} order_id={order_id} status={status} {message}")
    if str(status) == '57':
        _log_error(C, f"{stock} 委托被拒绝/junk order_id={order_id}: {message}")
        _send_wechat_notification(
            C,
            f'【@all 委托拒绝】{stock}',
            f'@all\norder_id={order_id}\n状态=57（拒绝/junk）\n原因={message}',
        )


def deal_callback(C, deal_info):
    """记录原生 QMT 成交回报，仓位在下一次决策时以券商实际数据同步。"""
    stock = _callback_value(deal_info, 'stock_code', 'stock', default='未知标的')
    volume = _callback_value(deal_info, 'volume', 'trade_volume', 'business_volume', default=0)
    price = _callback_value(deal_info, 'price', 'trade_price', 'business_price', default=0)
    direction = _callback_value(deal_info, 'order_type', 'direction', default='未知方向')
    _log(C, LOG_INFO, 'ORDER',
         f"成交回报 stock={stock} direction={direction} volume={volume} price={price}")


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def handlebar(C):
    now = datetime.now()
    if C.trade_mode == 'auto':
        _submit_due_pending_orders(C, now)

    if C.trade_mode != 'backtest' and not C.is_last_bar():
        return

    C._bar_callback_count = getattr(C, '_bar_callback_count', 0) + 1
    notify_period = _status_notification_period(now)

    if C.trade_mode in ('notify', 'auto') and not notify_period:
        return

    idx, idx_prev = _signal_index(C, now)
    _, time_str = _parse_bar_time(C, idx + 1)
    status_messages = []

    for stock in C.stock_list:
        try:
            _process_one(C, stock, time_str, idx, idx_prev, status_messages)
        except Exception as e:
            _log_error(C, f"{stock} 处理异常: {type(e).__name__}: {e}")
            _inc(C, 'error')
            continue
        
    if (C.trade_mode in ('notify', 'auto')
            and status_messages
            and notify_period
            and not _already_sent_daily_status(time_str, notify_period)):
        notification_sent = _send_wechat_notification(
            C,
            f'{notify_period}日线状态 {time_str}',
            '\n'.join(status_messages),
        )
        if notification_sent:
            _mark_daily_status_sent(time_str, notify_period)
            _upload_log_to_github(C)
        else:
            _log_warn(C, f"{notify_period} {time_str} 日线状态未发送成功，本时段将重试")
            
    if C._bar_callback_count % 50 == 0:
        _log(C, LOG_INFO, 'STAT', f"运行统计: {getattr(C, 'stats', {})}")


def _process_one(C, stock, time_str, idx, idx_prev, status_messages):
    fields = ['close', 'high', 'low', 'volume', 'amount']
    end_time = ''
    if time_str and time_str != '----':
        end_time = time_str.replace('-', '')
    data = _get_market_data_safe(C, stock, fields, count=120, end_time=end_time)
    if data is None or data.get('close') is None or len(data['close']) < 60:
        _log(C, LOG_DEBUG, 'SKIP', f"{stock} 数据不足，跳过")
        _inc(C, 'skip_data')
        return

    close = data['close']
    high = data.get('high')
    low = data.get('low')
    volume = data.get('volume')
    amount = data.get('amount')

    if amount is not None and len(amount) == len(close):
        amt = amount
    elif volume is not None and len(volume) == len(close):
        amt = volume * close
    else:
        amt = None

    n = len(close)
    if idx < -n or idx_prev < -n:
        return

    dif, dea, macd_hist = talib.MACD(close, C.macd_fast, C.macd_slow, C.macd_signal)
    rsi = talib.RSI(close, timeperiod=C.rsi_period)
    slowk, slowd, j = _kdj_tdx(high, low, close, C.kdj_n, C.kdj_m1, C.kdj_m2)
    adx = talib.ADX(high, low, close, timeperiod=C.adx_period)
    atr = talib.ATR(high, low, close, timeperiod=C.atr_period)

    curr_close = float(close[idx])
    curr_dif = float(dif[idx]) if not np.isnan(dif[idx]) else None
    curr_dea = float(dea[idx]) if not np.isnan(dea[idx]) else None
    prev_dif = float(dif[idx_prev]) if not np.isnan(dif[idx_prev]) else None
    prev_dea = float(dea[idx_prev]) if not np.isnan(dea[idx_prev]) else None
    curr_hist = float(macd_hist[idx]) if not np.isnan(macd_hist[idx]) else None
    prev_hist = float(macd_hist[idx_prev]) if not np.isnan(macd_hist[idx_prev]) else None
    curr_rsi = float(rsi[idx]) if not np.isnan(rsi[idx]) else None
    prev_rsi = float(rsi[idx_prev]) if not np.isnan(rsi[idx_prev]) else None
    curr_k = float(slowk[idx]) if not np.isnan(slowk[idx]) else None
    prev_k = float(slowk[idx_prev]) if not np.isnan(slowk[idx_prev]) else None
    curr_d = float(slowd[idx]) if not np.isnan(slowd[idx]) else None
    prev_d = float(slowd[idx_prev]) if not np.isnan(slowd[idx_prev]) else None
    curr_j = float(j[idx]) if not np.isnan(j[idx]) else None
    curr_adx = float(adx[idx]) if not np.isnan(adx[idx]) else None
    curr_atr = float(atr[idx]) if not np.isnan(atr[idx]) else None

    if any(v is None for v in [curr_dif, curr_dea, curr_hist, prev_hist, curr_rsi, curr_k, curr_d]):
        _log(C, LOG_DEBUG, 'SKIP', f"{stock} 指标含NaN，跳过")
        return

    vol_ratio = None
    vol_ok = True
    if amt is not None and len(amt) >= C.volume_ma_period + 5:
        amt_ma = talib.SMA(amt.astype(float), timeperiod=C.volume_ma_period)
        curr_amt = float(amt[idx])
        curr_amt_ma = float(amt_ma[idx]) if not np.isnan(amt_ma[idx]) else None
        if curr_amt_ma and curr_amt_ma > 0:
            vol_ratio = curr_amt / curr_amt_ma
            vol_ok = vol_ratio >= C.volume_mult
        else:
            vol_ok = True
    else:
        vol_ok = True

    regime = 'mid'
    if curr_adx is not None:
        if curr_adx > C.adx_trend:
            regime = 'trend'
        elif curr_adx < C.adx_range:
            regime = 'range'

    macd_green_shrinking = prev_hist < 0 and curr_hist < 0 and curr_hist > prev_hist
    kdj_golden = (prev_k is not None and prev_d is not None
                  and prev_k <= prev_d and curr_k > curr_d)
    kdj_oversold = (curr_k < C.kdj_k_oversold
                    or (curr_j is not None and curr_j < C.kdj_j_oversold))
    kdj_ok = kdj_golden and kdj_oversold
    rsi_ok = (curr_rsi <= C.rsi_oversold
              and prev_rsi is not None and curr_rsi > prev_rsi)
    mode = getattr(C, 'regime_mode', 'auto')
    if mode == 'kdj':
        entry_trigger, trigger_by = kdj_ok, 'KDJ'
    elif mode == 'rsi':
        entry_trigger, trigger_by = rsi_ok, 'RSI'
    elif mode == 'both':
        entry_trigger = kdj_ok or rsi_ok
        trigger_by = 'KDJ' if kdj_ok else ('RSI' if rsi_ok else '')
    elif regime == 'range':
        entry_trigger, trigger_by = kdj_ok, 'KDJ(震荡)'
    elif regime == 'trend':
        entry_trigger, trigger_by = rsi_ok, 'RSI(趋势)'
    else:
        entry_trigger, trigger_by = kdj_ok and rsi_ok, 'KDJ|RSI(中间)'

    if mode == 'auto':
        regime_description = {
            'range': '震荡市（ADX<20，使用KDJ金叉且超卖）',
            'trend': '趋势市（ADX>25，使用RSI超卖回升）',
            'mid': '中间状态（20≤ADX≤25，KDJ与RSI需同时满足）',
        }[regime]
    elif mode == 'kdj':
        regime_description = '强制KDJ模式（ADX仅供参考）'
    elif mode == 'rsi':
        regime_description = '强制RSI模式（ADX仅供参考）'
    else:
        regime_description = 'KDJ或RSI模式（ADX仅供参考）'

    st = _ensure_pos_state(C, stock)
    try:
        broker_pos = _get_broker_position_volume(C, stock)
    except Exception as error:
        _log_error(C, f"{stock} 查询券商持仓失败，跳过本次决策: {error}")
        return
    if st['vol'] > 0 and broker_pos <= 0:
        _reset_position_state(st)
        _log(C, LOG_INFO, 'POS', f"{stock} 已从券商持仓同步为无持仓")
    elif st['vol'] <= 0 and broker_pos > 0:
        st['vol'] = int(broker_pos)
        st['buy_price'] = curr_close
        st['buy_time'] = time_str + '(RECOVER)'
        st['high_since_entry'] = curr_close
        st['buy_count'] = 1
        _log(C, LOG_WARN, 'POS', f"{stock} 从券商恢复持仓 vol={st['vol']}")
    elif broker_pos > 0 and st['vol'] != broker_pos:
        st['vol'] = int(broker_pos)
        _log(C, LOG_INFO, 'POS', f"{stock} 已从券商持仓同步 vol={st['vol']}")

    has_pos = st['vol'] > 0

    if has_pos:
        st['bars_held'] = st.get('bars_held', 0) + 1
        st['high_since_entry'] = max(float(st.get('high_since_entry') or 0), curr_close)

    can_buy = (not has_pos) or (
        C.enable_repeat_buy and st.get('buy_count', 0) < C.max_buy_count
    )
    vol_info = f"量比={vol_ratio:.2f}" if vol_ratio is not None else "量比=N/A"
    vol_flag = "满足放量" if vol_ok else "不满足放量"
    atr_str = f"{curr_atr:.4f}" if curr_atr is not None else "N/A"
    j_str = f"{curr_j:.1f}" if curr_j is not None else "N/A"
    detail_payload = (
        f"close={curr_close:.3f} "
        f"DIF={curr_dif:.4f} DEA={curr_dea:.4f} MACD柱={curr_hist:.4f} "
        f"RSI={curr_rsi:.1f} K={curr_k:.1f} D={curr_d:.1f} J={j_str} "
        f"ADX={curr_adx:.1f}({regime}) ATR={atr_str} "
        f"{vol_info}({vol_flag}) pos={st['vol']}\n"
        f"ADX状态：{regime_description}；"
        f"买入条件：MACD绿柱缩短={'是' if macd_green_shrinking else '否'} "
        f"({prev_hist:.4f}→{curr_hist:.4f})；"
        f"KDJ金叉={'是' if kdj_golden else '否'}、超卖={'是' if kdj_oversold else '否'}；"
        f"RSI超卖回升={'是' if rsi_ok else '否'}；"
        f"{'放量过滤=关闭' if not C.use_volume_filter else f'放量={vol_flag}'}；"
        f"可买入仓位={'是' if can_buy else '否'}；"
        f"策略触发={trigger_by if entry_trigger else '否'}；"
    )
    status_messages.append(f"{stock} {detail_payload}")

    if (C.log_level >= LOG_DETAIL and getattr(C, 'log_mod_bar', True)
            and _should_log_detail_bar(stock, time_str)):
        _log(C, LOG_DETAIL, 'BAR', f"{stock} {time_str} {detail_payload}")

    if has_pos:
        sell_reason = None
        sell_ratio = 0.0

        buy_price = float(st.get('buy_price') or 0)
        pnl = _pnl_pct(buy_price, curr_close)
        high_since = float(st.get('high_since_entry') or curr_close)

        if C.use_stop_loss:
            if C.stop_mode == 'pct':
                if pnl is not None and pnl <= -C.stop_loss_pct:
                    sell_reason = f"固定止损({pnl*100:.2f}%)"
                    sell_ratio = 1.0
            else:
                entry_atr = float(st.get('entry_atr') or curr_atr or 0)
                if entry_atr > 0 and curr_close <= buy_price - C.stop_atr_mult * entry_atr:
                    sell_reason = "ATR止损"
                    sell_ratio = 1.0

        if sell_reason is None and C.use_take_profit:
            if C.stop_mode == 'pct':
                if pnl is not None and pnl >= C.take_profit_pct:
                    sell_reason = f"固定止盈({pnl*100:.2f}%)"
                    sell_ratio = 1.0
            else:
                entry_atr = float(st.get('entry_atr') or curr_atr or 0)
                if entry_atr > 0 and curr_close >= buy_price + C.take_atr_mult * entry_atr:
                    sell_reason = "ATR止盈"
                    sell_ratio = 1.0

        if sell_reason is None and C.use_trailing and high_since > 0:
            drawdown = (high_since - curr_close) / high_since
            if drawdown >= C.trail_pct:
                sell_reason = f"移动止盈(从高点回撤{drawdown*100:.2f}%)"
                sell_ratio = 1.0

        if sell_reason is None:
            div_end = None if idx == -1 else n + idx + 1
            if _detect_macd_top_divergence(close[:div_end],
                                           dif[:div_end],
                                           lookback=C.div_lookback,
                                           peak_order=C.div_peak_order):
                sell_reason = "MACD顶背离"
                sell_ratio = 1.0

        if sell_reason is None:
            death = (prev_dif is not None and prev_dea is not None and
                     prev_dif > prev_dea and curr_dif < curr_dea)
            if death:
                sell_reason = "MACD死叉"
                sell_ratio = float(C.death_cross_sell_ratio)

        if sell_reason is None and C.use_time_stop:
            if st.get('bars_held', 0) >= C.max_hold_days:
                sell_reason = f"时间止损(持仓{st['bars_held']}天)"
                sell_ratio = 1.0

        pnl_str = f"盈亏={pnl*100:+.2f}%" if pnl is not None else "盈亏=N/A"
        status_messages.append(
            f"{stock} 卖出条件：{pnl_str}；"
            f"止损={sell_reason if sell_reason and '止损' in sell_reason else ('关闭' if not C.use_stop_loss else '未触发')}；"
            f"止盈={sell_reason if sell_reason and '止盈' in sell_reason else ('关闭' if not C.use_take_profit else '未触发')}；"
            f"移动止盈={sell_reason if sell_reason and '移动止盈' in sell_reason else ('关闭' if not C.use_trailing else '未触发')}；"
            f"顶背离={'是' if sell_reason == 'MACD顶背离' else '否'}；"
            f"死叉={'是' if sell_reason == 'MACD死叉' else '否'}；"
            f"时间止损={sell_reason if sell_reason and '时间止损' in sell_reason else ('关闭' if not C.use_time_stop else '未触发')}；"
            f"最终判断={sell_reason or '不卖出'}"
        )

        if sell_reason and sell_ratio > 0:
            if _already_signaled(C, stock, time_str, f'SELL_{sell_reason}'):
                return
            sell_vol = int(st['vol'] * sell_ratio)
            sell_vol = (sell_vol // 100) * 100
            if sell_vol < 100 and st['vol'] >= 100:
                sell_vol = 100 if sell_ratio >= 0.5 else 0
            if sell_vol <= 0:
                return

            _log(C, LOG_INFO, 'SIGNAL',
                 f"【{time_str} 卖出】{stock} {sell_reason} 比例={sell_ratio:.0%} "
                 f"数量={sell_vol} 成本={buy_price:.3f} 现价={curr_close:.3f} {pnl_str}")

            if C.trade_mode == 'notify':
                status_messages.append(
                    f"{stock} 【卖出信号】原因={sell_reason} 数量={sell_vol} "
                    f"现价={curr_close:.3f} {pnl_str} 请人工确认后下单。"
                )
                if SEND_SIGNAL_NOTIFICATIONS:
                    _send_wechat_notification(
                        C,
                        f'卖出信号 {stock}',
                        f'时间={time_str}\n原因={sell_reason}\n数量={sell_vol}\n'
                        f'现价={curr_close:.3f}\n盈亏={pnl_str}\n请人工确认后下单。',
                    )
                _inc(C, 'signal_sell')
                return

            if C.trade_mode == 'auto':
                order = {
                    'signal_date': time_str,
                    'stock': stock,
                    'side': 'SELL',
                    'volume': sell_vol,
                    'reason': sell_reason,
                    'reference_price': curr_close,
                }
                try:
                    queued = _queue_pending_order(C, order)
                except Exception as error:
                    _log_error(C, f"{stock} 卖出信号未写入待执行队列: {error}")
                    return
                status_messages.append(
                    f"{stock} 【卖出信号】原因={sell_reason} 数量={sell_vol} "
                    f"收盘价={curr_close:.3f} {pnl_str}；"
                    f"将在下一交易日 {_qmt_time(C.auto_execution_time)} 后自动委托。"
                )
                if queued:
                    _log(C, LOG_INFO, 'SIGNAL',
                         f"{stock} 卖出信号已加入待执行队列 vol={sell_vol}")
                    _inc(C, 'signal_sell')
                return

            if C.submit_orders:
                try:
                    passorder(24, 1101, C.account, stock, 14, -1, sell_vol, # type: ignore
                              'MACD_KDJ_RSI', 1, '', C)
                    _log(C, LOG_INFO, 'ORDER', f"{stock} 卖出委托已提交 vol={sell_vol}")
                    _inc(C, 'order_sell')
                except Exception as e:
                    _log_error(C, f"{stock} 卖出下单失败，持仓状态未更新: {e}")
                    return

            st['vol'] -= sell_vol
            if st['vol'] <= 0:
                st['vol'] = 0
                st['buy_price'] = 0.0
                st['bars_held'] = 0
                st['high_since_entry'] = 0.0
                st['buy_count'] = 0
            _inc(C, 'signal_sell')
            return
    else:
        status_messages.append(f"{stock} 卖出条件：无持仓，不检查卖出信号。")

    if can_buy:
        macd_ok = macd_green_shrinking
        trigger = entry_trigger

        if trigger and C.use_volume_filter and not vol_ok:
            vol_r_str = f"{vol_ratio:.2f}" if vol_ratio is not None else "N/A"
            _log(C, LOG_DEBUG, 'DECISION',
                 f"{stock} 信号被成交量过滤 量比={vol_r_str} < {C.volume_mult}")
            trigger = False
            _inc(C, 'filter_volume')

        if not macd_ok:
            if C.log_level >= LOG_DEBUG:
                _log(C, LOG_DEBUG, 'DECISION',
                     f"{stock} MACD未满足 绿色柱连续为负且缩短={macd_green_shrinking} "
                     f"(前柱={prev_hist:.4f}, 当前柱={curr_hist:.4f})")
            trigger = False

        if trigger and macd_ok:
            if _already_signaled(C, stock, time_str, 'BUY'):
                return

            vol = _calc_volume_by_amount(C.buy_amount, curr_close)
            if vol < 100:
                _log(C, LOG_WARN, 'SIGNAL', f"{stock} 计算出的买入数量不足100，跳过")
                return

            j_str = f"{curr_j:.1f}" if curr_j is not None else "N/A"
            vol_r_str = f"{vol_ratio:.2f}" if vol_ratio is not None else "N/A"
            _log(C, LOG_INFO, 'SIGNAL',
                 f"【{time_str} {'加仓' if has_pos else '买入'}】{stock} 触发={trigger_by} "
                 f"MACD绿色柱缩短={prev_hist:.4f}→{curr_hist:.4f} "
                 f"RSI={curr_rsi:.1f} K={curr_k:.1f} J={j_str} "
                 f"ADX={curr_adx:.1f}({regime}) 量比={vol_r_str} "
                 f"买入数量={vol} 约{vol*curr_close:.0f}元 "
                 f"次数={st.get('buy_count', 0) + 1}/{C.max_buy_count}")

            if C.trade_mode == 'notify':
                status_messages.append(
                    f"{stock} 【{'加仓' if has_pos else '买入'}信号】触发={trigger_by} "
                    f"数量={vol} 现价={curr_close:.3f} "
                    f"MACD柱={prev_hist:.4f}→{curr_hist:.4f} 请人工确认后下单。"
                )
                if SEND_SIGNAL_NOTIFICATIONS:
                    _send_wechat_notification(
                        C,
                        f"{'加仓' if has_pos else '买入'}信号 {stock}",
                        f'时间={time_str}\n触发={trigger_by}\n数量={vol}\n'
                        f'现价={curr_close:.3f}\nMACD柱={prev_hist:.4f}→{curr_hist:.4f}\n'
                        f'次数={st.get("buy_count", 0) + 1}/{C.max_buy_count}\n请人工确认后下单。',
                    )
                _inc(C, 'signal_buy')
                return

            if C.trade_mode == 'auto':
                order = {
                    'signal_date': time_str,
                    'stock': stock,
                    'side': 'BUY',
                    'volume': vol,
                    'reason': trigger_by,
                    'reference_price': curr_close,
                    'entry_atr': curr_atr if curr_atr else 0.0,
                }
                try:
                    queued = _queue_pending_order(C, order)
                except Exception as error:
                    _log_error(C, f"{stock} 买入信号未写入待执行队列: {error}")
                    return
                status_messages.append(
                    f"{stock} 【{'加仓' if has_pos else '买入'}信号】触发={trigger_by} "
                    f"数量={vol} 收盘价={curr_close:.3f}；"
                    f"将在下一交易日 {_qmt_time(C.auto_execution_time)} 后自动委托。"
                )
                if queued:
                    _log(C, LOG_INFO, 'SIGNAL',
                         f"{stock} 买入信号已加入待执行队列 vol={vol}")
                    _inc(C, 'signal_buy')
                return

            if C.submit_orders:
                try:
                    passorder(23, 1101, C.account, stock, 14, -1, vol, # type: ignore
                              'MACD_KDJ_RSI', 1, '', C)
                    _log(C, LOG_INFO, 'ORDER', f"{stock} 买入委托已提交 vol={vol}")
                    _inc(C, 'order_buy')
                except Exception as e:
                    _log_error(C, f"{stock} 买入下单失败，持仓状态未更新: {e}")
                    return

            if has_pos:
                old_vol = st['vol']
                st['vol'] = old_vol + vol
                st['buy_price'] = (st['buy_price'] * old_vol + curr_close * vol) / st['vol']
                st['high_since_entry'] = max(st['high_since_entry'], curr_close)
            else:
                st['vol'] = vol
                st['buy_price'] = curr_close
                st['buy_time'] = time_str
                st['bars_held'] = 0
                st['high_since_entry'] = curr_close
                st['entry_atr'] = curr_atr if curr_atr else 0.0
            st['buy_count'] = st.get('buy_count', 0) + 1
            _inc(C, 'signal_buy')

        elif C.log_level >= LOG_DEBUG and macd_ok:
            reasons = []
            if mode in ('kdj', 'auto') and regime == 'range' and not kdj_ok:
                reasons.append(f"KDJ未满足(金叉={kdj_golden},超卖={kdj_oversold})")
            if mode in ('rsi', 'auto') and regime == 'trend' and not rsi_ok:
                reasons.append(f"RSI未满足(RSI={curr_rsi:.1f})")
            if mode == 'both' and not (kdj_ok or rsi_ok):
                reasons.append("KDJ与RSI均未满足")
            if reasons:
                _log(C, LOG_DEBUG, 'DECISION', f"{stock} 未买入: {'; '.join(reasons)}")
    elif C.log_level >= LOG_DEBUG and has_pos:
        _log(C, LOG_DEBUG, 'DECISION',
             f"{stock} 不再加仓: enable_repeat_buy={C.enable_repeat_buy} "
             f"buy_count={st.get('buy_count', 0)}/{C.max_buy_count}")


# =============================================================================
# 结束
# =============================================================================
