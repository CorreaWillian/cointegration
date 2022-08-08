import pandas as pd
from cointegracao import Cointegration
from itertools import combinations, permutations
import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map # or thread_map
from copy import deepcopy

class Executer(Cointegration):

    def __init__(self, bd, permut, coint, train_size=252, twoway=False, moving_limits=False):

        self.bd = bd
        self.permut = permut
        self.coint = coint
        self.coint_temp = deepcopy(coint)
        self.train_size = train_size
        self.twoway = twoway
        self.moving_limits = moving_limits


    def backtest(self, pair):

        """
        Run backtest of a pair over pre setted period
        Returns dictionary with results
        """
        
        first_stock, scnd_stock = pair
        test_size = len(self.bd) - self.train_size

        # Initiate the variables
        status = False
        open_price_first_stock = None
        open_price_scnd_stock = None
        open_date = None
        std_open_residual = None
        residual_open=None
        beta_open = None
        beta_close = None
        close_limit = stop_limit = None
        halflife = None
        days_open = 0
        op_id = -1

        results_dict = []

        # Loop over test_size
        for i in range(test_size):
            
            # Slices the dataframe until 'period' i and selects lasts 'train_size' observations
            test = self.bd.iloc[ : self.train_size + i][[first_stock, scnd_stock]][-self.train_size:]
            
            # If exist missing values in any stock test next period
            if test.isna().any().any():
                return []

            # Check if there's a open position
            if not status:
                
                # Cointegration test. Returns True or False
                coint_test = self.coint.cointegration_test(first_stock=test[first_stock], scnd_stock=test[scnd_stock])

                # Checks if pair is meets the requirements to open position
                if coint_test and self.coint.check_open():
                    
                    # If True, verify if the inverse pair meets the conditions to open position
                    if self.twoway:
                        coint_temp_test = self.coint_temp.cointegration_test(first_stock=test[scnd_stock], scnd_stock=test[first_stock])
                        if coint_temp_test and self.coint_temp.check_open():
                            pass
                        else: 
                            return

                    # Variables to annotate
                    open_price_first_stock = test[first_stock].iloc[-1]
                    open_price_scnd_stock = test[scnd_stock].iloc[-1]
                    open_date = test.index[-1]
                    residual_open = self.coint.residuals.iloc[-1]
                    std_open_residual = self.coint.residuals.std()
                    beta_open = self.coint.beta
                    close_limit, stop_limit = self.coint.close_limits()             
                    status = True
                    halflife = self.coint.halflife()

                else:
                    continue

            # If there's a open position
            else:

                days_open += 1

                self.coint.regression(test[first_stock], test[scnd_stock])

                # Movin Limits
                if self.moving_limits:
                    close_limit, stop_limit = self.coint.close_limits() 
                #Checks if pair meets the requirements to close position
                if self.coint.check_close(close_limit, stop_limit, days_open):
                    
                    # Set status as close and annotate the beta
                    status = 'close'
                    beta_close = self.coint.beta            

            # Create the dictionary with results
            results_dict.append({
                # 'op_id': op_id,
                'date': test.index[-1],
                'pair': pair,
                'status': status,
                'price_fst_stock': test[first_stock].iloc[-1],
                'price_scnd_stock': test[scnd_stock].iloc[-1],
                'beta_open': beta_open,
                'beta_close': beta_close,
                'last_residual': self.coint.residuals.iloc[-1],
                'std_residual': self.coint.residuals.std(),
                'std_open_residual': std_open_residual,
                'residual_open': residual_open,
                'open_price_first_stock': open_price_first_stock,
                'open_price_scnd_stock': open_price_scnd_stock,
                'open_date': open_date,
                'close_limit': close_limit,
                'stop_limit': stop_limit,
                'halflife': halflife,
                'days_open': days_open
                })

            # Reinitiate the variables
            if status == 'close':

                status = False
                op_id = -1
                open_price_first_stock = None
                open_price_scnd_stock = None
                open_date = None
                std_open_residual = None
                residual_open=None
                beta_open = None
                beta_close = None
                close_limit = None
                stop_limit = None
                halflife = None
                days_open = 0

        return results_dict


    def executer(self, n_workers=7):
        
        """
        Executes the backtest for multiple pairs in
        multithreading.
        """

        results = process_map(self.backtest, self.permut, max_workers=n_workers, chunksize=1)

        return results
 

# df.loc[df.pair == ('ARZZ3', 'AZUL4')]

# start = time.time()

# results = executer()
# print(time.time() - start)
# print(results)
