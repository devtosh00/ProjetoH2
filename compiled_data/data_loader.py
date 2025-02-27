import os
import pandas as pd

def load_csv(file_path, parse_dates=True, index_col=0):
    return pd.read_csv(file_path, index_col=index_col, parse_dates=parse_dates)

def load_all_data(base_folder, start, end):
    # Carrega os dados dos ativos
    gld = load_csv(os.path.join(base_folder, "GLD_daily.csv"))
    iau = load_csv(os.path.join(base_folder, "IAU_daily.csv"))
    paxg = load_csv(os.path.join(base_folder, "PAXG_daily.csv"))
    xauusd = load_csv(os.path.join(base_folder, "XAUUSD_daily.csv"))
    fred = load_csv(os.path.join(base_folder, "fred_interest_rates.csv"))
    
    # Filtra pelo período
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    gld = gld[(gld.index >= start) & (gld.index < end)]
    iau = iau[(iau.index >= start) & (iau.index < end)]
    paxg = paxg[(paxg.index >= start) & (paxg.index < end)]
    xauusd = xauusd[(xauusd.index >= start) & (xauusd.index < end)]
    fred = fred[(fred.index >= start) & (fred.index < end)]
    
    return gld, iau, paxg, xauusd, fred
