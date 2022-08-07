from arch import arch_model
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from arch.unitroot import engle_granger, ADF

from matplotlib import pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from itertools import combinations, permutations
import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map # or thread_map

class Cointegration:

    def __init__(self, significance=0.05, constant=True, z_score_in=2, z_score_out=0.5, z_score_stop=1):

        self.significance = significance
        self.constant = constant
        self.z_score_in = z_score_in
        self.z_score_out = z_score_out
        self.z_score_stop = z_score_stop


    def adf(self, col):

        """
        Check if series is I(1)
        Calculates dick-fulley test in level and first difference
        to check if series is not stationary in level but is in first difference.
        Returns True if pass the test.
        """

        # ADF test in level
        adf_level = ADF(col, method='BIC')

        # ADF test in first difference
        adf_first_diff = ADF(np.diff(col), method='BIC')

        # Check if is not stationary in level but is in first difference
        if adf_level.pvalue > self.significance and self.significance > adf_first_diff.pvalue:
            # Return True if pass
            return True
        else:
            return False


    def regression(self, first_stock, scnd_stock):

        """
        Series is cointegrated if regression residuals are stationary
        Returns True for cointegrated or False if not cointegrated
        """

        self.first_stock = first_stock
        self.scnd_stock = scnd_stock
        
        Y = first_stock
        X = scnd_stock

        # If wants to add constant term to regression
        if self.constant:
            X = sm.add_constant(X)

        # Initiates the model and fit
        model = sm.OLS(Y, X)
        self.reg = model.fit()

        # Gets beta of regression
        self.beta = self.reg.params[-1]

        self.residuals = self.reg.resid


    def cointegration_test(self, first_stock, scnd_stock):

        self.first_stock = first_stock
        self.scnd_stock = scnd_stock
        
        """
        Series is cointegrated if regression residuals are stationary
        Returns True for cointegrated or False if not cointegrated
        """

        # Check if stocks are I(1)
        if not (self.adf(first_stock) and self.adf(scnd_stock)):
            return False

        """
        self.regression(first_stock, scnd_stock)

        # Performs dickey-fuller test in regression residuals
        adf_resid = ts.adfuller(self.residuals)

        # Gets pvalue of dickey-fuller test
        resid_pvalue = adf_resid[1]
        """

        arch_coint = engle_granger(first_stock, scnd_stock)
        self.residuals = arch_coint.resid
        self.beta = - arch_coint.cointegrating_vector[1]

        # Returs True if pvalue is lower than significance and False if not
        if arch_coint.pvalue < self.significance:
            return True
        else:
            return False

    
    def halflife(self):

        """
        Calculates half life for mean reverting.
        """

        # Regression residual first difference
        residual_lag = self.residuals.diff().fillna(method='bfill')
        residual_lag = sm.add_constant(residual_lag)
        # print(residual_lag)
        # print('-'*30)
        # print(self.residuals)
        # Regression residuals with residual_lag
        model = sm.OLS(residual_lag, self.residuals)
        res_reg = model.fit()

        self.half_life = (round(np.log(2) / res_reg.params[1])).item()
        
        return self.half_life
        
    
    def check_open(self):

        """
        Checks if pair meets the requirements to open position
        """

        self.limit_in = self.residuals.mean() + (self.residuals.std() * self.z_score_in)

        if abs(self.residuals.iloc[-1]) > self.limit_in:
            return True
        else:
            return False
    

    def close_limits(self):

        """
        Create the requirements to close position
        """

        close_limit =  self.residuals.mean() + (self.residuals.std() * self.z_score_out)
        stop_limit = self.residuals.mean() + ((self.residuals.std() * self.z_score_in) + self.z_score_stop)
        
        return close_limit, stop_limit


    def check_close(self, close_limit, stop_limit, days_open):

        """
        Checks if pair meets the requirements to close position
        """
        # close_limit =  self.residuals.mean() + (self.residuals.std() * self.z_score_out)
        # stop_limit = self.residuals.mean() + ((self.residuals.std() * self.z_score_in) + self.z_score_stop)

        if  (close_limit > abs(self.residuals.iloc[-1])) or (abs(self.residuals.iloc[-1]) > stop_limit) or (days_open > self.half_life):
            return True
        else:
            return False


    def plot_scatter(self):

        """
        Plots the scatterplot of stocks and its fitted regression line
        """
        sns.scatterplot(x=self.scnd_stock, y=self.first_stock)
        sns.lineplot(x=self.scnd_stock, y=self.reg.fittedvalues, color='red', label='Regression')
        
        plt.xlabel(self.scnd_stock.name)
        plt.ylabel(self.first_stock.name)
        plt.legend()

        plt.show()

    
    def plot_resid_bounds(self):

        """
        Plots the residual lineplot with upper and lower bounds
        defined by z-score.
        """

        upper = self.residuals.std() * self.z_score_in
        lower = self.residuals.std() * - self.z_score_in

        sns.lineplot(x=self.residuals.index, y=self.residuals)

        plt.axhline(y=self.residuals.mean(), color='red', linestyle='--', linewidth=1, label='Mean')
        plt.axhline(y=upper, color='blue', linestyle='--', linewidth=1, label=f'+/-{self.z_score_in} z-score')
        plt.axhline(y=lower, color='blue', linestyle='--', linewidth=1, label='_nolegend_')

        plt.xlabel('Date')
        plt.ylabel('Residuals')

        plt.legend()
        plt.show()
        

    def plot_prices(self):

        """
        Plot the pair historical prices
        """

        sns.lineplot(y=self.first_stock, x=self.first_stock.index, label=self.first_stock.name)
        sns.lineplot(y=self.scnd_stock, x=self.scnd_stock.index, label=self.scnd_stock.name)

        plt.xlabel('Date')
        plt.ylabel('Prices')

        plt.show()

