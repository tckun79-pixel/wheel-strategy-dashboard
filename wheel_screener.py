import math
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Union

def get_real_market_data(ticker: str) -> Dict[str, Union[str, float, int, dict]]:
    """
    Fetches real market data using yfinance for strategy analysis.
    """
    ticker = ticker.upper()
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        price = info.last_price
        
        # Estimate IV (Simplified as yfinance doesn't provide IVR directly without heavy processing)
        # We'll use a placeholder or simplified calculation if needed, 
        # but for now let's focus on price and option chains.
        
        # Get options expiration dates
        expirations = tk.options
        if not expirations:
            return None
            
        # Target ~45 DTE
        target_date = datetime.now() + timedelta(days=45)
        best_exp = min(expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))
        
        opt_chain = tk.option_chain(best_exp)
        puts = opt_chain.puts
        calls = opt_chain.calls
        
        # Quantitative Delta Parameters
        is_high_priced = price >= 100
        target_delta = 0.18 if is_high_priced else 0.30
        
        # Find best CSP strike (closest to target delta)
        puts['delta_diff'] = abs(puts['delta'].abs() - target_delta) if 'delta' in puts.columns else abs(puts['strike'] - (price * 0.95))
        best_put = puts.loc[puts['delta_diff'].idxmin()]
        
        # Find best CC strike (approx 15% OTM or delta 0.25)
        calls['delta_diff'] = abs(calls['delta'].abs() - 0.25) if 'delta' in calls.columns else abs(calls['strike'] - (price * 1.10))
        best_call = calls.loc[calls['delta_diff'].idxmin()]
        
        # Calculate DTE
        dte = (datetime.strptime(best_exp, '%Y-%m-%d') - datetime.now()).days

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "ivr": 50, # Placeholder as IVR requires historical data
            "iv": best_put.get('impliedVolatility', 0.5) * 100,
            "dte": dte,
            "csp": {
                "strike": best_put['strike'],
                "premium": best_put['lastPrice'],
                "delta": best_put.get('delta', -target_delta),
            },
            "cc": {
                "strike": best_call['strike'],
                "premium": best_call['lastPrice'],
                "delta": best_call.get('delta', 0.25),
            }
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def analyze_strategy_optimized(tickers: List[str], max_capital: float) -> Dict:
    """
    Applies the Wheel Strategy formulas with real data.
    """
    results = []
    total_capital_required = 0.0
    
    for t in tickers:
        data = get_real_market_data(t.strip())
        if not data:
            continue
            
        capital_req = data["csp"]["strike"] * 100
        dte = max(1, data["dte"])

        # 1. Cash-Secured Put (CSP) ROC
        csp_roc = (data["csp"]["premium"] / (data["csp"]["strike"] - data["csp"]["premium"])) * (365 / dte) if data["csp"]["strike"] > data["csp"]["premium"] else 0
        
        # 2. Full Loop Blended ROC
        total_profit = data["csp"]["premium"] + data["cc"]["premium"] + (data["cc"]["strike"] - data["csp"]["strike"])
        blended_roc = (total_profit / data["csp"]["strike"]) * (365 / (dte * 2)) if data["csp"]["strike"] > 0 else 0

        # 3. Defensive Rolling (Stress Scenario)
        btc_cost = data["csp"]["premium"] * 3.5
        new_strike = data["csp"]["strike"] - (5 if data["price"] > 100 else 2)
        net_credit = 1.00 if data["price"] > 100 else 0.40 # Simplified net credit projection

        # 4. Stop Loss Metrics (200% Net Loss)
        stop_trigger = data["csp"]["premium"] * 3
        defined_net_loss = data["csp"]["premium"] - stop_trigger
        capital_protected = ((capital_req - abs(defined_net_loss * 100)) / capital_req) * 100 if capital_req > 0 else 0

        results.append({
            "ticker": data["ticker"],
            "price": data["price"],
            "ivr": data["ivr"],
            "capital_req": capital_req,
            "csp_strike": data["csp"]["strike"],
            "csp_premium": data["csp"]["premium"],
            "csp_roc": csp_roc * 100,
            "cc_strike": data["cc"]["strike"],
            "cc_premium": data["cc"]["premium"],
            "blended_roc": blended_roc * 100,
            "defense_roll_strike": new_strike,
            "defense_net_credit": net_credit,
            "stop_loss_loss": defined_net_loss * 100,
            "capital_protected": capital_protected
        })
        total_capital_required += capital_req

    results.sort(key=lambda x: x["csp_roc"], reverse=True)

    return {
        "results": results,
        "total_capital": total_capital_required,
        "max_exceeded": total_capital_required > max_capital
    }
