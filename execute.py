import pandas as pd
from cointegracao import Cointegration
from itertools import combinations, permutations
import time
from backtest import Executer
from tqdm.contrib.concurrent import process_map, thread_map # or thread_map



bd = pd.read_excel('BD COMPLETO.xlsx')
bd.dropna(axis=0, inplace=True)
bd.set_index('Data', inplace=True)
bd = bd.loc['2018-02-27': '2019-03-02']

bd = bd.apply(pd.to_numeric)

coint = Cointegration(significance = 0.01, z_score_in=2)

i_1 = [coint.adf(bd[col]) for col in bd.columns]
i_1 = [ ele for ele in i_1 if ele is not None ]

permut = list(combinations(i_1, 2))

