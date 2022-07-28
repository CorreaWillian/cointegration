import pandas as pd
from cointegracao import Cointegration
from itertools import combinations, permutations
import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map # or thread_map

global bd, coint, permut
bd = pd.read_excel('BD COMPLETO.xlsx')
bd.dropna(axis=0, inplace=True)
bd.set_index('Data', inplace=True)
bd = bd.loc['2018-02-27': '2019-03-02']

coint = Cointegration(significance = 0.01, z_score_in=2)

i_1 = [coint.adf(bd[col]) for col in bd.columns]
i_1 = [ ele for ele in i_1 if ele is not None ]

permut = list(combinations(i_1, 2))


def cointegrated(pair):
    try:
        coint_test = coint.cointegration_test(first_stock=bd[pair[0]], scnd_stock=bd[pair[1]])
        if coint_test:
            return pair
    except:
        pass

def executar(permut):
    if __name__ == '__main__':

        # with Pool(7) as p:
        #     results = p.imap(cointegrated, permut)
        results = process_map(cointegrated, permut, max_workers=7, chunksize=1)
        results = [ ele for ele in results if ele is not None]

        return results



# start = time.time()

# results = executar()
# print(time.time() - start)
# print(results)
