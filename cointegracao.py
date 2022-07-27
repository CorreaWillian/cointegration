import pandas as pd
import statsmodels.tsa.stattools as ts
from itertools import combinations, permutations
import time
import numpy as np
import matplotlib.pyplot as plt
from arch.unitroot import engle_granger
from datetime import datetime
import xlwings as xw
from functools import reduce
import operator
from hurst import compute_Hc

#df = pd.read_excel(r'C:\Users\Administrator\Desktop\back_tests\2DP_BETA.xlsx', index_col=0)
bd = pd.read_excel(r'C:\Users\Administrator\Desktop\back_tests\HISTORICO ETFS.xlsx', sheet_name='Planilha2', index_col=0)
# wb = xw.Book(r'C:\Users\willi\OneDrive\GAP\python\RTD.xlsx').sheets[1].range("A1").expand('table').options(pd.DataFrame).value
# df = df.append(wb[0:1], sort=True).dropna(axis=1)


def adf(col):
    result = ts.adfuller(col, regression='ctt')
    result2 = ts.adfuller(np.diff(col), regression='ctt')
    if result[1] > 0.1 > result2[1]:
        return col.name


def aprovado(prc_long, prc_short, par, beta, stat, H, residuo, desv_pad, data, long, short, invertido):
    d = {'Data': data,
         'Long': long,
         'Short': short,
         'Preço Long': prc_long,
         'Preço Short': prc_short,
         'Comprar': 'OK',
         'Venda Long': '',
         'Venda Short': '',
         'Data Fechamento': '',
         'Desvio Padrão': desv_pad,
         'Residuo': residuo[-1],
         'Beta': beta,
         'Stat': stat,
         'Hurst': H,
         'Acao 1': par[0],
         'Acao 2': par[1],
         'Total': '',
         'Invertido': invertido,
         'Passou $1': None, 'Data $1': None, 'Passou $2': None, 'Data $2': None, 'Passou $3': None, 'Data $3': None
         }
    # arch_coint.plot(title=par[0]+'x'+par[1])
    # plt.axhline(1.8 * desv_pad, color='red')
    # plt.axhline(-1.8 * desv_pad, color='red')
    # plt.axhline(color='green')
    # plt.show()
    return d


def cointegracao(par, first_col, scnd_col, data, desvios=1):

    arch_coint = engle_granger(first_col, scnd_col, trend='ctt', lags=1)
    if arch_coint.pvalue < 0.01:
        residuo = list(arch_coint.resid)
        desv_pad = np.std(residuo)
        beta = arch_coint._xsection.params[0]
        stat = arch_coint.stat
        H, c, val = compute_Hc(residuo)
        if residuo[-1] > desvios*desv_pad and beta > 0:
            d = aprovado(scnd_col[-1], first_col[-1], par, beta, stat, H,
                         residuo, desv_pad, data, long=par[1], short=par[0], invertido='Sim')
            return d
        elif residuo[-1] < -desvios*desv_pad and beta > 0:
            d = aprovado(first_col[-1], scnd_col[-1], par, beta, stat, H,
                         residuo, desv_pad, data, long=par[0], short=par[1], invertido='Nao')
            return d
        else:
            return {}
    else:
        return {}


def monitorar(df, bd, intervalo):
    # print(bd)
    data = bd.index[-1]
    for index, row in df.iterrows():
        if row['Comprar'] == 'OK':
            y = bd[row['Acao 1']].values[-intervalo:]
            x = bd[row['Acao 2']].values[-intervalo:]
            results = engle_granger(y, x, trend='ctt', lags=1)
            residuo = list(results.resid)
            beta = row['Beta']

            venda_long = bd[row['Long']].values[-1]
            venda_short = bd[row['Short']].values[-1]
            if row['Invertido'] == 'Nao':
                total = (- row['Preço Long'] + venda_long) + (row['Preço Short'] - venda_short) * beta
            else:
                total = (- row['Preço Long'] + venda_long) * beta + (row['Preço Short'] - venda_short)

            df.at[index, 'Venda Long'] = venda_long
            df.at[index, 'Venda Short'] = venda_short

            if 2 > total >= 1 and (row['Passou $1'] is None or str(row['Passou $1']) == 'nan'):
                df.at[index, 'Passou $1'] = total
                df.at[index, 'Data $1'] = data
            elif 3 > total >= 2 and (row['Passou $2'] is None or str(row['Passou $2']) == 'nan'):
                df.at[index, 'Passou $2'] = total
                df.at[index, 'Data $2'] = data
            elif total >= 3 and (row['Passou $3'] is None or str(row['Passou $3']) == 'nan'):
                df.at[index, 'Passou $3'] = total
                df.at[index, 'Data $3'] = data

            if row['Invertido'] == 'Sim' and residuo[-1] < 0:
            # if row['Invertido'] == 'Sim' and (residuo[-1] < 0 or total > 0.25):
            # if row['Invertido'] == 'Sim' and total > 0.50:
                df.at[index, 'Comprar'] = 'Vender'
                df.at[index, 'Data Fechamento'] = data

            elif row['Invertido'] == 'Nao' and residuo[-1] > 0:
            # elif row['Invertido'] == 'Nao' and (residuo[-1] > 0 or total > 0.25):
            # elif row['Invertido'] == 'Nao' and  total > 0.50:
                df.at[index, 'Comprar'] = 'Vender'
                df.at[index, 'Data Fechamento'] = data

            df.at[index, 'Total'] = total
    return df


def par_aberto(row):
    return [row['Acao 1'], row['Acao 2']]


def executar(df, bd, intervalo=220):

    boas = bd[-intervalo:].apply(lambda col: adf(col), axis=0)
    boas = list(filter(None.__ne__, boas))
    boas = list(permutations(boas, 2))
    if df.empty is False:
        df_abertos = df[df['Comprar'] == 'OK']
        if df_abertos.empty is False:
            df = monitorar(df, bd, intervalo)
            abertos = list(df_abertos.apply(lambda row: (row['Acao 1'], row['Acao 2']), axis=1))
            print(abertos)
            abertos = list(filter(None.__ne__, abertos))
            abertos = [list(permutations(x)) for x in abertos]
            abertos = (reduce(operator.add, abertos))
            print(abertos)
            print(len(abertos))
            boas = list(set(boas) - set(abertos))
    print(boas)
    print(len(boas))
    data = bd.index[-1]
    resultados = [cointegracao(colname,
                  bd[colname[0]].to_numpy(),
                  bd[colname[1]].to_numpy(),
                  data) for colname in boas]

    df_novo = pd.DataFrame(resultados)
    # df_novo.set_index('Data', inplace=True)
    df_novo.dropna(how='all', inplace=True)

    df = df.append(df_novo, ignore_index=True)
    df.to_excel(r'C:\Users\Administrator\Desktop\back_tests\1DP_ctt_stat_lag1_H_BETA_ETFS_META_VARIOS.xlsx')
    return df


df = pd.DataFrame()
for i, (index, row) in enumerate(bd.iterrows()):
    if i > 220:
        antes = time.time()
        df = executar(df, bd[:i])
        print(time.time() - antes)
