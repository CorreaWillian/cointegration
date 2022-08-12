import pandas as pd
import numpy as np
from cointegracao import Cointegration
from itertools import combinations, permutations
import time
from backtest import Executer
from tqdm.contrib.concurrent import process_map, thread_map # or thread_map

import gc

df_prices = pd.read_excel('database.xlsx', index_col=0)
portfolio = pd.read_excel('PORTFOLIO2.xlsx')

portfolio.replace('VIVT4', 'VIVT3',inplace=True)
portfolio = portfolio[portfolio.ticker.isin(df_prices.columns)]

def create_request(df):
    # Permutation over same sectors stocks

    # Initiates a dictionary with key as initial date 
    permut = {data_ini:[data_fin, []] for data_ini, data_fin in zip(df.data_ini.unique(), df.data_fin.unique())}

    # Loop over initial dates
    for data in df.data_ini.unique():   
        
        # Select stocks with same initial date
        df_sem = df.loc[df.data_ini==data]
        
        # Loops over sectors to create a permutation
        for setor in df_sem.setor.unique():
            
            # Select stocks of same sector and appends its permutations on dictionary
            tickers = df_sem.loc[df_sem.setor == setor, 'ticker'].to_list()
            permut[data][-1].extend(list(permutations(tickers, 2)))
    
    return permut

requests = create_request(portfolio)

coint = Cointegration(z_score_out=0.5, z_score_stop=1000, conf_var=0.05)

train_size = 252
# Loops over dict with initial, final dates and permutations
# And filters the dataframe with prices with 252 days (one year)
# of formation data and 6 months of trading data
dic_list = []

for key, value in requests.items():
    
    data_ini = key
    data_pre =  data_ini - np.timedelta64(500, 'D')
    data_fin = value[0]

    index_ini = df_prices.index.get_indexer(target=[key], method='ffill').item()
    index_pre = df_prices.index.get_indexer(target=[data_ini], method='ffill').item() - train_size
    index_fin = df_prices.index.get_indexer(target=[value[0]], method='ffill').item()

    perm = value[1]

    corr_df = df_prices[:index_ini][-train_size: ]
    correlations = {}
    for pair in perm:
        correlations[pair] = corr_df[list(pair)].corr().iloc[0][1]

    last_quartile_corr = pd.DataFrame([correlations]).T.sort_values(by=0).quantile(0.75)

    coint.min_corr = last_quartile_corr.item()
    # coint.min_corr = 0.9

    # df_trading = df_prices.iloc[index_pre: index_fin]

    
    # train_size = df_trading[data_pre:data_ini].shape[0] + 1        

    e = Executer(bd=df_prices, idx_start_date=index_ini, idx_end_date=index_fin, permut=perm, coint=coint, train_size=train_size, twoway=False, moving_limits=False)

    if __name__ == '__main__':
        dic_list.extend(e.executer())
        del e
        gc.collect()