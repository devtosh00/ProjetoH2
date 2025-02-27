import os
import pandas as pd

def adjust_scales(gld, iau, paxg):
    # Ajusta os preços para refletir o preço por onça
    gld["price_oz"] = gld["close"] * 10
    iau["price_oz"] = iau["close"] * 100
    paxg["price_oz"] = paxg["close"]
    return gld, iau, paxg

def align_data(gld, iau, paxg, xauusd, fred, start, end):
    # Alinha os dados usando a interseção dos índices
    common_dates = gld.index.intersection(iau.index).intersection(paxg.index).intersection(xauusd.index)
    gld = gld.loc[common_dates]
    iau = iau.loc[common_dates]
    paxg = paxg.loc[common_dates]
    xauusd = xauusd.loc[common_dates]
    
    # Para o FRED, reindexa para dias úteis e preenche para frente
    business_dates = pd.date_range(start, end, freq='B')
    fred = fred.sort_index().reindex(business_dates, method='ffill')
    # Converte taxa para retorno diário (usando 'close')
    fred["risk_free"] = (1 + fred["close"] / 100) ** (1/252) - 1
    fred_risk = fred["risk_free"].reindex(common_dates, method='ffill')
    
    return gld, iau, paxg, xauusd, fred_risk, common_dates

def create_unified_df(gld, iau, paxg, xauusd, fred_risk):
    df = pd.DataFrame(index=gld.index)
    df["GLD"] = gld["price_oz"]
    df["IAU"] = iau["price_oz"]
    df["PAXG"] = paxg["price_oz"]
    df["XAUUSD"] = xauusd["close"]
    df["avg_etf"] = df[["GLD", "IAU", "PAXG"]].mean(axis=1)
    df["spread"] = df["avg_etf"] - df["XAUUSD"]
    df = df.join(fred_risk)
    return df
