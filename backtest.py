import pandas as pd
from cointegracao import Cointegration
from itertools import combinations, permutations
import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map # or thread_map


class Executer(Cointegration):

    def __init__(self, bd, permut, coint):

        self.bd = bd
        self.permut = permut
        self.coint = coint

    def backtest(self, pair, train_size=182):

        first_stock, scnd_stock = pair
        test_size = len(self.bd) - train_size

        status = False

        results_dict = []

        open_price_first_stock = None
        open_price_scnd_stock = None
        open_date = None
        std_open_residual = None
        residual_open=None

        op_id = -1

        for i in range(test_size):

            test = self.bd.iloc[i: train_size + i, ][[first_stock, scnd_stock]]
            
            if not status:

                coint_test = self.coint.cointegration_test(first_stock=test[first_stock], scnd_stock=test[scnd_stock])
                
                if coint_test and self.coint.check_open():
                    
                    open_price_first_stock = test[first_stock].iloc[-1]
                    open_price_scnd_stock = test[scnd_stock].iloc[-1]
                    open_date = test.index[-1]
                    residual_open = self.coint.residuals.iloc[-1]
                    std_open_residual = self.coint.residuals.std()

                    status = True

            else:

                self.coint.regression(test[first_stock], test[scnd_stock])
                
                if self.coint.check_close():

                    status = 'close'

            results_dict.append({
                # 'op_id': op_id,
                'date': self.coint.residuals.index[-1],
                'pair': pair,
                'status': status,
                'price_fst_stock': test[first_stock].iloc[-1],
                'price_scnd_stock': test[scnd_stock].iloc[-1],
                'beta': self.coint.beta,
                'last_residual': self.coint.residuals.iloc[-1],
                'std_residual': self.coint.residuals.std(),
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


    def executer(self, n_workers=7):
        
        results = process_map(self.backtest, self.permut, max_workers=n_workers, chunksize=1)

        return results
 

# df.loc[df.pair == ('ARZZ3', 'AZUL4')]

# start = time.time()

# results = executer()
# print(time.time() - start)
# print(results)
