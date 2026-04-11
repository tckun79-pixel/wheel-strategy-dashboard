import math
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Union
import streamlit as st
import traceback


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_historical_close(ticker: str) -> pd.Series:
    """Cached 3‑month closing price history for HV30 computation."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="3mo")
    if hist.empty:
        return pd.Series(dtype=float)
    return hist["Close"]


def get_real_market_data(ticker: str, csp_target_delta: float = None, cc_target_delta: float = None) -> Dict[str, Union[str, float, int, dict]]:
    """
    Fetches real market data using yfinance for strategy analysis.
    csp_target_delta: desired absolute delta for CSP (default auto: 0.18 for high-price, 0.30 for low-price).
    cc_target_delta:  desired delta for covered call (default 0.25).
    Returns None if no options data available.
    """
    ticker = ticker.upper()
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        price = info.last_price
        if price is None or price <= 0:
            print(f"Invalid price for {ticker}: {price}")
            return None
            
        # Get options expiration dates
        expirations = list(tk.options)
        if not expirations:
            print(f"[{ticker}] No options expirations available")
            return None

        # Target ~45 DTE, try up to 3 closest expirations until we get valid chain
        target_date = datetime.now() + timedelta(days=45)
        sorted_expirations = sorted(expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))
        
        best_put = None
        best_call = None
        chosen_exp = None
        
        for exp in sorted_expirations[:3]:  # Try up to 3 closest expirations
            try:
                opt_chain = tk.option_chain(exp)
                puts = opt_chain.puts
                calls = opt_chain.calls
                
                if puts.empty or calls.empty:
                    print(f"[{ticker}] Empty chain for {exp} (puts={len(puts)}, calls={len(calls)}), trying next expiry...")
                    continue
                    
                # We have data! Pick strikes.
                chosen_exp = exp
                
                # Quantitative Delta Parameters
                is_high_priced = price >= 100
                csp_delta_target_val = csp_target_delta if csp_target_delta is not None else (0.18 if is_high_priced else 0.30)
                cc_delta_target_val = cc_target_delta if cc_target_delta is not None else 0.25
                
                # Find best CSP strike
                if 'delta' in puts.columns and not puts.empty:
                    puts_copy = puts.copy()
                    puts_copy['delta_diff'] = abs(puts_copy['delta'].abs() - csp_delta_target_val)
                    best_put = puts_copy.loc[puts_copy['delta_diff'].idxmin()]
                else:
                    # Fallback: 5% OTM put
                    target_strike = price * 0.95
                    puts_copy = puts.copy()
                    puts_copy['strike_diff'] = abs(puts_copy['strike'] - target_strike)
                    best_put = puts_copy.loc[puts_copy['strike_diff'].idxmin()]
                
                # Find best CC strike
                if 'delta' in calls.columns and not calls.empty:
                    calls_copy = calls.copy()
                    calls_copy['delta_diff'] = abs(calls_copy['delta'].abs() - cc_delta_target_val)
                    best_call = calls_copy.loc[calls_copy['delta_diff'].idxmin()]
                else:
                    # Fallback: 10% ITM call
                    target_strike = price * 1.10
                    calls_copy = calls.copy()
                    calls_copy['strike_diff'] = abs(calls_copy['strike'] - target_strike)
                    best_call = calls_copy.loc[calls_copy['strike_diff'].idxmin()]
                
                # Success — break out of loop
                break
                
            except Exception as e:
                print(f"[{ticker}] Error processing expiry {exp}: {e}")
                continue
        
        if best_put is None or best_call is None:
            print(f"[{ticker}] Could not find valid strikes across top 3 expirations")
            return None
        
        # Calculate DTE from chosen expiry
        try:
            dte = max(1, (datetime.strptime(chosen_exp, '%Y-%m-%d') - datetime.now()).days)
        except Exception:
            dte = 30
        
        # Compute implied volatility and IVR vs 30‑day historical volatility
        iv = best_put.get('impliedVolatility', 0.5)
        if iv is None or iv <= 0:
            iv = 0.5
        iv = iv * 100  # convert to percentage
        
        try:
            close = _fetch_historical_close(ticker)
            if not close.empty and len(close) >= 30:
                returns = close.pct_change().dropna()
                hv30_std = returns[-30:].std() * math.sqrt(250) * 100  # annualized
                hv30 = hv30_std
            else:
                hv30 = None
        except Exception as e:
            print(f"HV calc error for {ticker}: {e}")
            hv30 = None
        
        if hv30 and hv30 > 0:
            ivr = (iv / hv30) * 100
        else:
            ivr = 0.0
        
        return {
            "ticker": ticker,
            "price": round(price, 2),
            "ivr": ivr,
            "iv": iv,
            "dte": dte,
            "csp": {
                "strike": best_put['strike'],
                "premium": best_put.get('lastPrice', 0),
                "delta": best_put.get('delta', -csp_delta_target_val),
            },
            "cc": {
                "strike": best_call['strike'],
                "premium": best_call.get('lastPrice', 0),
                "delta": best_call.get('delta', cc_delta_target_val),
            }
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def analyze_strategy_optimized(tickers: List[str], max_capital: float, csp_target_delta: float = None, cc_target_delta: float = None) -> Dict:
    """
    Applies the Wheel Strategy formulas with real data.
    """
    results = []
    total_capital_required = 0.0
    
    for t in tickers:
        t_clean = t.strip()
        if not t_clean:
            continue
            
        data = get_real_market_data(t_clean, csp_target_delta=csp_target_delta, cc_target_delta=cc_target_delta)
        if not data:
            print(f"Skipping {t_clean}: no data")
            continue
            
        try:
            capital_req = data["csp"]["strike"] * 100
            dte = max(1, data["dte"])

            # 1. Cash-Secured Put (CSP) ROC
            csp_premium = data["csp"]["premium"]
            csp_strike = data["csp"]["strike"]
            if csp_strike > csp_premium and csp_strike > 0:
                csp_roc = (csp_premium / (csp_strike - csp_premium)) * (365 / dte)
            else:
                csp_roc = 0
            
            # 2. Full Loop Blended ROC
            cc_premium = data["cc"]["premium"]
            cc_strike = data["cc"]["strike"]
            total_profit = csp_premium + cc_premium + (cc_strike - csp_strike)
            if csp_strike > 0:
                blended_roc = (total_profit / csp_strike) * (365 / (dte * 2))
            else:
                blended_roc = 0

            # 3. Defensive Rolling (Stress Scenario)
            btc_cost = csp_premium * 3.5
            new_strike = csp_strike - (5 if data["price"] > 100 else 2)
            net_credit = 1.00 if data["price"] > 100 else 0.40

            # 4. Stop Loss Metrics (200% Net Loss)
            stop_trigger = csp_premium * 3
            defined_net_loss = csp_premium - stop_trigger
            capital_protected = ((capital_req - abs(defined_net_loss * 100)) / capital_req) * 100 if capital_req > 0 else 0

            results.append({
                "ticker": data["ticker"],
                "price": data["price"],
                "ivr": data["ivr"],
                "iv": data["iv"],
                "dte": dte,
                "capital_req": capital_req,
                "csp_strike": csp_strike,
                "csp_premium": csp_premium,
                "csp_roc": csp_roc * 100,
                "csp_delta": data["csp"]["delta"],
                "csp_delta_abs": abs(data["csp"]["delta"]),
                "cc_strike": cc_strike,
                "cc_premium": cc_premium,
                "cc_delta": data["cc"]["delta"],
                "blended_roc": blended_roc * 100,
                "defense_roll_strike": new_strike,
                "defense_net_credit": net_credit,
                "stop_loss_loss": defined_net_loss * 100,
                "capital_protected": capital_protected
            })
            total_capital_required += capital_req
        except Exception as e:
            print(f"Error processing {t}: {e}")
            continue

    results.sort(key=lambda x: x["csp_roc"], reverse=True)

    return {
        "results": results,
        "total_capital": total_capital_required,
        "max_exceeded": total_capital_required > max_capital
    }
