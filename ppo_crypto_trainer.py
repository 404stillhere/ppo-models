"""
PPO Crypto Trainer — BTC/ETH/SOL/BNB
Train: bybit hourly candles → MaskablePPO
"""
import os, sys, time, traceback
from datetime import datetime, timezone

sys.path.insert(0, "M:/temp_downloads")
os.chdir("C:/OpenMythos")

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv("M:/temp_downloads/.env")

# ─── CONFIG ─────────────────────────────────────────────────────────────────
TICKERS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
WINDOW     = 60
COMMISSION = 0.0005   # Bybit spot
BYBIT_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET", "")
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  PPO Crypto Trainer — BTC/ETH/SOL/BNB")
print("=" * 60)
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print()

# Force UTF-8
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─── DATA DOWNLOAD ────────────────────────────────────────────────────────────

def download_crypto(ticker, limit=5000):
    """Download hourly candles from Bybit."""
    try:
        client = HTTP(api_key=BYBIT_KEY, api_secret=BYBIT_SECRET)
        all_rows = []
        cursor = None

        # Bybit max limit per request is 1000
        kwargs = {"category": "linear", "symbol": ticker, "interval": 60, "limit": 1000}
        resp = client.get_kline(**kwargs)
        if resp.get("retCode") != 0:
            print(f"    [!] {ticker}: {resp['retMsg']}")
            return None

        rows = resp["result"]["list"]
        for r in rows:
            ts_ms = int(r[0])
            if ts_ms > 1e12:
                ts_ms //= 1000
            o, h, l, c, v = map(float, r[1:6])
            all_rows.append({
                "ts": datetime.fromtimestamp(ts_ms, tz=timezone.utc),
                "open": o, "high": h, "low": l, "close": c, "volume": v,
            })

        cursor = resp["result"].get("nextPageCursor")
        # Fetch next pages if needed and available
        while cursor and len(all_rows) < limit:
            resp = client.get_kline(category="linear", symbol=ticker,
                                    interval=60, limit=200, cursor=cursor)
            if resp.get("retCode") != 0:
                break
            rows = resp["result"]["list"]
            if not rows:
                break
            for r in rows:
                ts_ms = int(r[0])
                if ts_ms > 1e12:
                    ts_ms //= 1000
                o, h, l, c, v = map(float, r[1:6])
                all_rows.append({
                    "ts": datetime.fromtimestamp(ts_ms, tz=timezone.utc),
                    "open": o, "high": h, "low": l, "close": c, "volume": v,
                })
            cursor = resp["result"].get("nextPageCursor")
            if not cursor:
                break

        df = pd.DataFrame(all_rows[::-1]).reset_index(drop=True)
        s = f"  {ticker}: {len(df)} bars | {df['ts'].iloc[0].date()} - {df['ts'].iloc[-1].date()}"
        print(s)
        return df
    except Exception as e:
        print(f"  [!] {ticker} download failed: {e}")
        return None


# ─── INDICATORS ──────────────────────────────────────────────────────────────

def compute_all(close, high, low, window=WINDOW):
    n = len(close)
    # RSI-14 (vectorized)
    deltas = np.diff(close, prepend=close[0])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.convolve(gains,  np.ones(14)/14, mode="same")
    avg_loss = np.convolve(losses, np.ones(14)/14, mode="same")
    rs  = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-period)
    sma20 = np.convolve(close, np.ones(20)/20, mode="same")
    std20 = np.array([close[max(0,i-20):i+1].std() for i in range(n)])
    bb_up = sma20 + 2 * std20
    bb_lo = sma20 - 2 * std20

    # ATR-14
    tr = np.zeros(max(0, n-1))
    for i in range(1, n):
        tr[i-1] = max(high[i], close[i-1]) - min(low[i], close[i-1])
    atr = np.convolve(tr, np.ones(14)/14, mode="same")

    # Returns
    pct_ret = np.concatenate([[0], np.diff(close) / (close[:-1] + 1e-10)])

    return rsi, bb_up, bb_lo, atr, pct_ret


# ─── ENV ──────────────────────────────────────────────────────────────────────

class CryptoEnv(gym.Env):
    """
    obs = [rsi/100, bb_pos, atr_pct, vol, position] + returns_window(WINDOW)
    act = 0 (HOLD), 1 (BUY/close)
    """
    def __init__(self, close, high, low, rsi, bb_up, bb_lo, atr, rets,
                 window=WINDOW, commission=COMMISSION):
        super().__init__()
        self.close  = close.astype(np.float32)
        self.high   = high.astype(np.float32)
        self.low    = low.astype(np.float32)
        self.rsi    = rsi.astype(np.float32)
        self.bb_up  = bb_up.astype(np.float32)
        self.bb_lo  = bb_lo.astype(np.float32)
        self.atr    = atr.astype(np.float32)
        self._rets  = rets.astype(np.float32)
        self.window = window
        self.commission = commission
        self.n = n = len(close)

        # volatility
        self._vol = np.array([
            np.std(rets[max(0,i-20):i]) * np.sqrt(365)
            for i in range(n)
        ], dtype=np.float32)

        self.max_step     = n - window - 1
        self.current_step = window
        self.position     = 0
        self.entry_price  = 0.0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window + 5,), dtype=np.float32)

    def _get_obs(self):
        i = self.current_step
        ret_win = self._rets[i - self.window:i]
        rsi_v = float(self.rsi[i] if not np.isnan(self.rsi[i]) else 50.0) / 100
        bb_p  = float((self.close[i] - self.bb_lo[i]) /
                      (self.bb_up[i] - self.bb_lo[i] + 1e-10))
        atr_p = float(self.atr[i] / (self.close[i] + 1e-10))
        vol_v = float(self._vol[i])
        pos   = 1.0 if self.position == 1 else 0.0
        return np.concatenate([[rsi_v, bb_p, atr_p, vol_v, pos]], dtype=np.float32)

    def action_masks(self):
        return np.array([True, True], dtype=bool)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.position     = 0
        self.entry_price  = 0.0
        self.current_step = self.window
        return self._get_obs(), {}

    def step(self, action):
        i  = self.current_step
        c  = self.close[i]
        reward = 0.0

        # Close existing
        if self.position != 0:
            reward = (c - self.entry_price) / (self.entry_price + 1e-10) - self.commission
            self.position = 0
            self.entry_price = 0.0

        # Open long
        if action == 1:
            self.position    = 1
            self.entry_price = c

        self.current_step += 1
        done = self.current_step >= self.max_step
        return self._get_obs(), reward, done, False, {}


class MultiEnv(gym.Env):
    """Wrapper — randomly selects ticker each step (train only)."""
    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.n    = len(envs)
        self._active = 0
        self.observation_space = envs[0].observation_space
        self.action_space      = envs[0].action_space
        self._rng = np.random.default_rng(42)

    def reset(self, seed=None, **kwargs):
        self._active = int(self._rng.integers(0, self.n))
        obs, info = self.envs[self._active].reset(seed=seed)
        return np.array(obs).astype(np.float32), info

    def step(self, action):
        self._active = int(self._rng.integers(0, self.n))
        result = self.envs[self._active].step(action)
        obs, reward, done, trunc, info = result
        return np.array(obs).astype(np.float32), reward, done, trunc, info

    def action_masks(self):
        return self.envs[self._active].action_masks()


# ─── INFERENCE ───────────────────────────────────────────────────────────────

def run_agent(model, env, deterministic=True):
    obs, _ = env.reset()
    done   = False
    cum    = 0.0
    curr_pos, curr_entry = 0, 0.0
    trades = []

    while not done:
        mask = env.action_masks()
        act, _ = model.predict(np.array(obs), action_masks=mask, deterministic=deterministic)
        prev_pos = curr_pos

        obs, _, done, _, _ = env.step(int(act))

        if env.position != prev_pos:
            if prev_pos == 1:
                pnl = (env.close[env.current_step-1] - curr_entry) / curr_entry * 100
                cum += pnl
                trades.append({"type": "exit", "pnl": pnl/100})
            if env.position == 1:
                curr_pos   = 1
                curr_entry = env.close[env.current_step-1]
            else:
                curr_pos, curr_entry = 0, 0.0

    return cum, trades


def metrics(cum_pnl, trades):
    if not trades:
        return {"return": 0, "sharpe": 0, "max_dd": 0, "n": 0}
    pnls = [t["pnl"] for t in trades]
    win_rate = sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100
    return {
        "return": round(cum_pnl, 2),
        "sharpe": 0,  # would need equity series
        "max_dd": 0,
        "n": len(trades),
        "win_rate": round(win_rate, 1),
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

print("[1] Downloading crypto data from Bybit...")
all_data = {}
for ticker in TICKERS:
    df = download_crypto(ticker)
    if df is not None and len(df) >= WINDOW + 100:
        all_data[ticker] = df
    time.sleep(0.5)

print(f"\nDownloaded {len(all_data)}/{len(TICKERS)} tickers")

if not all_data:
    print("[!] No data downloaded. Exiting.")
    sys.exit(1)

# Build envs
print("\n[2] Building gym environments...")
envs_train = []
envs_val   = {}

for ticker, df in all_data.items():
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    rsi, bb_up, bb_lo, atr, rets = compute_all(close, high, low)

    # Train/val split: 80/20
    split = int(len(close) * 0.80)
    env_train = CryptoEnv(
        close[:split], high[:split], low[:split],
        rsi[:split], bb_up[:split], bb_lo[:split],
        atr[:split], rets[:split], window=WINDOW)
    envs_train.append(env_train)

    envs_val[ticker] = CryptoEnv(
        close[split:], high[split:], low[split:],
        rsi[split:], bb_up[split:], bb_lo[split:],
        atr[split:], rets[split:], window=WINDOW)

multi_env = MultiEnv(envs_train)
print(f"  obs_dim={multi_env.observation_space.shape[0]}, train_envs={len(envs_train)}")

# Train
print("\n[3] Training MaskablePPO (200k steps)...")
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

ppo = MaskablePPO(
    MaskableActorCriticPolicy, multi_env,
    policy_kwargs=dict(net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]),
    learning_rate=8e-5,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.02,
    verbose=1,
    seed=42,
)

ppo.learn(total_timesteps=200_000, progress_bar=True)
MODEL_OUT = "C:/OpenMythos/maskable_ppo_crypto_v2.zip"
ppo.save(MODEL_OUT)
print(f"\nModel saved: {MODEL_OUT}")

# Validate
print("\n[4] Per-ticker validation...")
total_return = 0
for ticker, env in envs_val.items():
    cum, trades = run_agent(ppo, env)
    m = metrics(cum, trades)
    total_return += m["return"]
    print(f"  {ticker:8s} | Ret={m['return']:>8.2f}% | Trades={m['n']:>3d} | WinRate={m['win_rate']:>5.1f}%")

print(f"\n  Avg Return: {total_return/len(envs_val):.2f}%")
print("\n[OK] Training complete!")
