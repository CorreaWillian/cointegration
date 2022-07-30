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

bd = bd.apply(pd.to_numeric)

coint = Cointegration(significance = 0.01, z_score_in=2)

i_1 = [coint.adf(bd[col]) for col in bd.columns]
i_1 = [ ele for ele in i_1 if ele is not None ]

permut = list(combinations(i_1, 2))


def cointegrated(pair):

    """
    Test two stocks for cointegration and return
    the pair ticker if cointegrated
    """

    try:
        coint_test = coint.cointegration_test(first_stock=bd[pair[0]], scnd_stock=bd[pair[1]])
        if coint_test:
            return pair
        else:
            return False
    except:
        pass


def backtest(pair, train_size=200):

    first_stock, scnd_stock = pair
    test_size = len(bd) - train_size

    status = False

    results_dict = []

    open_price_first_stock = None
    open_price_scnd_stock = None
    open_date = None
    std_open_residual = None
    residual_open=None

    op_id = -1

    for i in range(test_size):

        test = bd.iloc[i: train_size + i, ][[first_stock, scnd_stock]]
        
        if not status:

            coint_test = coint.cointegration_test(first_stock=test[first_stock], scnd_stock=test[scnd_stock])
            
            if coint_test and coint.check_open():
                
                open_price_first_stock = test[first_stock].iloc[-1]
                open_price_scnd_stock = test[scnd_stock].iloc[-1]
                open_date = test.index[-1]
                residual_open = coint.residuals.iloc[-1]
                std_open_residual = coint.residuals.std()

                status = True

        else:

            coint.regression(test[first_stock], test[scnd_stock])
            
            if coint.check_close():

                status = 'close'

        results_dict.append({
            # 'op_id': op_id,
            'date': coint.residuals.index[-1],
            'pair': pair,
            'status': status,
            'price_fst_stock': test[first_stock].iloc[-1],
            'price_scnd_stock': test[scnd_stock].iloc[-1],
            'beta': coint.beta,
            'last_residual': coint.residuals.iloc[-1],
            'std_residual': coint.residuals.std(),
            'std_open_residual': std_open_residual,
            'residual_open': residual_open,
            'open_price_first_stock': open_price_first_stock,
            'open_price_scnd_stock': open_price_scnd_stock,
            'open_date': open_date,

            })

        if status == 'close':

            status = False
            op_id = -1
            open_price_first_stock = None
            open_price_scnd_stock = None
            open_date = None
            std_open_residual = None
            residual_open=None

    return results_dict


def executer(permut=permut[:500], n_workers=7):

    # if __name__ == '__main__':

    # with Pool(7) as p:
    #     results = p.imap(cointegrated, permut)
    results = process_map(backtest, permut, max_workers=n_workers, chunksize=1)
    # results = [ ele for ele in results if ele is not None]

    return results
 

# df.loc[df.pair == ('ARZZ3', 'AZUL4')]

# start = time.time()

# results = executer(permut[:])
# print(time.time() - start)
# # print(results)

# df_list = [pd.DataFrame(d) for d in results]
# df = pd.concat(df_list)
