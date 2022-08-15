import pandas as pd
import numpy as np
from cointegracao import Cointegration
from itertools import combinations, permutations
import time
from backtest import Executer
from tqdm.contrib.concurrent import process_map, thread_map # or thread_map
import ast

import gc

def create_request(periodo):

    global df_prices, portfolio
    portfolio = pd.read_excel('PORTFOLIO2.xlsx')
    portfolio = portfolio.set_index('data_ini')['2020':'2020-01-01'].reset_index() 

    if periodo.lower() == 'diario':        
        df_prices = pd.read_excel('database.xlsx', index_col=0)
    
    elif periodo.lower() == 'minutos':

        df_prices = pd.read_excel('database15min.xlsx', index_col=0)
        df_prices.set_index('time', inplace=True)
        df_prices = df_prices.loc['2019-06-01': '2020-07-12']
        portfolio = portfolio.set_index('data_ini')['2020':'2020-01-01'].reset_index()    
    
    elif periodo.lower() == '1minutos':

        df_prices = pd.read_excel('database1min.xlsx', index_col=0)
        df_prices.set_index('time', inplace=True)
        df_prices = df_prices.loc['2019-06-01': '2020-07-12']
        portfolio = portfolio.set_index('data_ini')['2020':'2020-01-01'].reset_index()  
    
    elif periodo.lower() == '5minutos':

        df_prices = pd.read_excel('database5min.xlsx', index_col=0)
        df_prices.set_index('time', inplace=True)
        df_prices = df_prices.loc['2019-06-01': '2020-07-12']
        portfolio = portfolio.set_index('data_ini')['2020':'2020-01-01'].reset_index()  
        
        
    portfolio.replace('VIVT4', 'VIVT3',inplace=True)
    df = portfolio[portfolio.ticker.isin(df_prices.columns)]

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


def executa(
    n_workers, 
    periodo, 
    train_size, 
    significancia, 
    z_score_saida, 
    z_score_stop, 
    conf_var, 
    usar_halflife, 
    usar_var):

    if z_score_stop.lower() == 'desligado':
        z_score_stop = 10000

    requests = create_request(periodo)

    coint = Cointegration(
        significance=significancia, 
        z_score_out=z_score_saida, 
        z_score_stop=z_score_stop, 
        conf_var=conf_var, 
        usar_halflife=usar_halflife, 
        usar_var=usar_var
        )

    train_size = train_size
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
    

        e = Executer(bd=df_prices, idx_start_date=index_ini, idx_end_date=index_fin, permut=perm, coint=coint, train_size=train_size, twoway=False, moving_limits=False)

        print(f'Executando {pd.to_datetime(data_ini).date()}/{pd.to_datetime(data_fin).date()}')
        dic_list.extend(e.executer(n_workers=n_workers))
        del e
        gc.collect()
    
    return dic_list


def calcula_retornos(dic_list, custos=0.0071):

    # Transforma o dicionario de resultados em dataframe
    df_list = [pd.DataFrame(d) for d in dic_list]
    df = pd.concat(df_list)
    
    # Conserta os formatos
    df['date'] = pd.to_datetime(df.date)
    df['open_date'] = pd.to_datetime(df.open_date)
    df['days_open'] = df['days_open'].astype(float)

    # Coleta os setores do arquivo de portfolio
    df['stock1'] = [pair.split("'")[1] for pair in df.pair.astype(str)]
    df = df.merge(portfolio.drop_duplicates('ticker'), left_on='stock1', right_on='ticker',how='left')

    # Dropa colunas desnecessárias
    df.drop(columns=['stock1', 'ticker'], inplace=True)

    # Ordena os nomes dos pares com entrada invertida
    # df['pair'] = df.pair.apply(ast.literal_eval)
    df['sorted_pair'] = df.pair.apply(sorted).apply(tuple)


    df.drop_duplicates(subset=['date', 'sorted_pair'], keep='first',inplace=True)

    # Inicia o calculo dos rertornos

    df.loc[df['residual_open'] < df['std_open_residual'], 'side'] = 'lower'
    df.loc[df['residual_open'] > df['std_open_residual'], 'side'] = 'upper'

    i=1
    returns_list = []       
    closed = df.loc[df.status=='close']

    for row in closed.itertuples():

        historico_par = df.loc[(df.pair==row.pair) & (df.date.between(row.open_date,row.date))].copy()
        historico_par.set_index('date', inplace=True)
        historico_par['id'] = i
        
    #     Custo entrada: 1,22% + 0,1% = 1,31%
    #     Custo saida: 1,22% + 0,25% + 0,1% = 1,57% + Aluguel
        
        aluguel = (1.0143)**((row.days_open+2)/252) -1
        # custos = 0.0071

        try:
            if row.side == 'lower':
                
                historico_par['ratio_sem_custos'] = historico_par['price_fst_stock'] / historico_par['price_scnd_stock']
                
                historico_par.at[row.open_date, 'price_fst_stock'] = historico_par.at[row.open_date, 'price_fst_stock']
                
                # Entrada
                historico_par.at[row.open_date, 'price_fst_stock'] = historico_par.loc[row.open_date, 'price_fst_stock'] * (1+custos)
                historico_par.at[row.open_date, 'price_scnd_stock'] = historico_par.loc[row.open_date, 'price_scnd_stock'] * (1-custos)
                
                        
                # Saída
                historico_par.at[row.date, 'price_fst_stock'] = historico_par.loc[row.date, 'price_fst_stock'] * (1-custos)
                historico_par.at[row.date, 'price_scnd_stock'] = historico_par.loc[row.date, 'price_scnd_stock'] * (1+(custos+aluguel))

                historico_par['ratio'] = historico_par['price_fst_stock'] / historico_par['price_scnd_stock']
                
            else:
                
                historico_par['ratio_sem_custos'] = historico_par['price_scnd_stock'] / historico_par['price_fst_stock']
                
                historico_par.at[row.open_date, 'price_scnd_stock'] = historico_par.loc[row.open_date, 'price_scnd_stock']
                # Entrada
                historico_par.at[row.open_date, 'price_scnd_stock'] = historico_par.loc[row.open_date, 'price_scnd_stock'] * (1+0.0071)
                historico_par.at[row.open_date, 'price_fst_stock'] = historico_par.loc[row.open_date, 'price_fst_stock'] * (1-0.0071)

                # Saida
                historico_par.at[row.date, 'price_scnd_stock'] = historico_par.loc[row.date, 'price_scnd_stock'] * (1-0.0071)
                historico_par.at[row.date, 'price_fst_stock'] = historico_par.loc[row.date, 'price_fst_stock'] * (1+(0.0071+aluguel))

                historico_par['ratio'] = historico_par['price_scnd_stock'] / historico_par['price_fst_stock']
            
            
            historico_par['open_price_first_stock'] = historico_par.at[row.open_date, 'price_fst_stock']
            historico_par['open_price_scnd_stock'] = historico_par.at[row.open_date, 'price_scnd_stock']
            
            historico_par['return'] = (historico_par.ratio / historico_par.ratio.shift(1)) -1
            historico_par['retorno_acumulado'] = np.cumprod(1+historico_par['return']) -1
            
            historico_par['return_sem_custos'] = (historico_par.ratio_sem_custos / historico_par.ratio_sem_custos.shift(1)) -1
            historico_par['retorno_acumulado_sem_custos'] = (np.cumprod(1+historico_par['return_sem_custos']) -1)
            
            returns_list.append(historico_par)
            
        except Exception as e:
            pass        
        i+=1

    # Junta os retornos
    df_returns = pd.concat(returns_list)
    df_returns.reset_index(inplace=True)

    # Filtra retornos após 2019 se houver
    df_returns = df_returns.set_index('open_date').loc['2019-01-01':]
    df_returns.reset_index(inplace=True)

    return df_returns


def novo_retorno(df, custo_compra, custo_venda, intraday=False):

    # Inicia o calculo dos rertornos

    df.loc[df['residual_open'] < df['std_open_residual'], 'side'] = 'lower'
    df.loc[df['residual_open'] > df['std_open_residual'], 'side'] = 'upper'

    i=1
    returns_list = []       
    closed = df.loc[df.status=='close']

    for row in closed.itertuples():

        historico_par = df.loc[(df.pair==row.pair) & (df.date.between(row.open_date,row.date))].copy()
        historico_par.sort_values(by='date', inplace=True)
        historico_par.set_index('date', inplace=True)
        historico_par['id'] = i
        
    #     Custo entrada: 1,22% + 0,1% = 1,31%
    #     Custo saida: 1,22% + 0,25% + 0,1% = 1,57% + Aluguel
        
        days_open = (row.date.date() - row.open_date.date()).days
        if intraday:
            aluguel = (1.0143)**((days_open+2)/252) -1
        else:
            aluguel = (1.0143)**((row.days_open+2)/252) -1


        historico_par.at[row.open_date, 'price_fst_stock'] = historico_par.at[row.open_date,'open_price_first_stock']
        historico_par.at[row.open_date, 'price_scnd_stock'] = historico_par.at[row.open_date, 'open_price_scnd_stock']

        # Daytrade
        if row.date.date() == row.open_date.date():
            custo_venda = custo_compra
            aluguel = 0
        
        try:
            if row.side == 'lower':
                
                historico_par['ratio_sem_custos'] = historico_par['price_fst_stock'] / historico_par['price_scnd_stock']
                
                historico_par.at[row.open_date, 'price_fst_stock'] = historico_par.at[row.open_date, 'price_fst_stock']
                
                # Entrada
                historico_par.at[row.open_date, 'price_fst_stock'] = historico_par.loc[row.open_date, 'price_fst_stock'] * (1 + custo_compra)
                historico_par.at[row.open_date, 'price_scnd_stock'] = historico_par.loc[row.open_date, 'price_scnd_stock'] * (1 - custo_venda)
                
                        
                # Saída
                historico_par.at[row.date, 'price_fst_stock'] = historico_par.loc[row.date, 'price_fst_stock'] * (1 - custo_venda)
                historico_par.at[row.date, 'price_scnd_stock'] = historico_par.loc[row.date, 'price_scnd_stock'] * (1 + (custo_compra+aluguel))

                historico_par['ratio'] = historico_par['price_fst_stock'] / historico_par['price_scnd_stock']
                
            else:
                
                historico_par['ratio_sem_custos'] = historico_par['price_scnd_stock'] / historico_par['price_fst_stock']
                
                historico_par.at[row.open_date, 'price_scnd_stock'] = historico_par.loc[row.open_date, 'price_scnd_stock']
                # Entrada
                historico_par.at[row.open_date, 'price_scnd_stock'] = historico_par.loc[row.open_date, 'price_scnd_stock'] * (1 + custo_compra)
                historico_par.at[row.open_date, 'price_fst_stock'] = historico_par.loc[row.open_date, 'price_fst_stock'] * (1 - custo_venda)

                # Saida
                historico_par.at[row.date, 'price_scnd_stock'] = historico_par.loc[row.date, 'price_scnd_stock'] * (1 - custo_venda)
                historico_par.at[row.date, 'price_fst_stock'] = historico_par.loc[row.date, 'price_fst_stock'] * (1 + (custo_compra + aluguel))

                historico_par['ratio'] = historico_par['price_scnd_stock'] / historico_par['price_fst_stock']
            
            
            historico_par['open_price_first_stock'] = historico_par.at[row.open_date, 'price_fst_stock']
            historico_par['open_price_scnd_stock'] = historico_par.at[row.open_date, 'price_scnd_stock']
            
            historico_par['return'] = (historico_par.ratio / historico_par.ratio.shift(1)) -1
            historico_par['retorno_acumulado'] = np.cumprod(1+historico_par['return']) -1
            
            historico_par['return_sem_custos'] = (historico_par.ratio_sem_custos / historico_par.ratio_sem_custos.shift(1)) -1
            historico_par['retorno_acumulado_sem_custos'] = (np.cumprod(1+historico_par['return_sem_custos']) -1)
            
            returns_list.append(historico_par)
            
        except Exception as e:
            pass        
        i+=1

    # Junta os retornos
    df_returns = pd.concat(returns_list)
    df_returns.reset_index(inplace=True)

    # Filtra retornos após 2019 se houver
    df_returns = df_returns.set_index('open_date').loc['2019-01-01':]
    df_returns.reset_index(inplace=True)

    return df_returns