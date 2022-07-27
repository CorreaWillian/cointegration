import pandas as pd
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from matplotlib import pyplot as plt


class Cointegration:

    def __init__(self, significance=0.05, constant=True):

        self.significance = significance
        self.constant = constant


    def adf(self, col):

        """
        Check if series is I(1)
        Calculates dick-fulley test in level and first difference
        to check if series is not stationary in level but is in first difference.
        Returns stock ticker if pass the test.
        """

        # ADF test in level
        adf_level = ts.adfuller(col, regression='ctt')

        # ADF test in first difference
        adf_first_diff = ts.adfuller(np.diff(col), regression='ctt')

        # Check if is not stationary in level but is in first difference
        if adf_level[1] > self.significance > adf_first_diff[1]:
            # Return the stock ticker if pass the test
            return col.name


    def cointegration_test(self, pairname, first_stock, scnd_stock, data, z_score):

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

        residuals = self.reg.resid

        # Performs dickey-fuller test in regression residuals
        adf_resid = ts.adfuller(residuals)

        # Gets pvalue of dickey-fuller test
        resid_pvalue = adf_resid[1]

        # Returs True if pvalue is lower than significance and False if not
        if resid_pvalue < self.significance:
            return True
        else:
            return False


    def plot_scatter(self):

        plt.scatter(x=self.scnd_stock, y=self.first_stock)
        plt.plot(self.scnd_stock, self.reg.fittedvalues, color='red', label='Regression')
        
        plt.xlabel(self.scnd_stock.name)
        plt.ylabel(self.first_stock.name)
        plt.legend()

        plt.show()
        
