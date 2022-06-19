import pandas as pd
import numpy as np
import ffn


class Isin:
    def __init__(self, isin_code, since=None):
        self.isin_code = isin_code
        if since == None:
            self.since = '2022-01-02'
        else:
            self.since = since
        df = pd.read_csv('data/simulator-database.csv', delimiter=';')
        df.drop_duplicates(inplace=True)
        df['NAV_DATE']=pd.to_datetime(df['PRICING_DATE'], format='%d.%m.%Y')
        self.df = df.copy()
        self.nav = df.query('ISIN_CODE == @isin_code and NAV_DATE>= @since ').sort_values('NAV_DATE', ascending=True)
    
    def get_navs(self):
        return self.nav[['NAV_DATE','NAV']].drop_duplicates().sort_values('NAV_DATE', ascending=False)

    def get_relative_perf(self):
        first_nav = self.nav.head(1)['NAV']
        last_nav = self.nav.tail(1)['NAV']
        #return last_nav / first_nav
        return (np.array(last_nav)[0] / np.array(first_nav)[0] - 1)* 100

class Portfolio:
    def __init__(self, listisin, since):
        self.isins = listisin
        self.since = since

    def create_series(self):
        isins = self.isins
        df = pd.read_csv('data/simulator-database.csv', delimiter=';')
        df.drop_duplicates(inplace=True)
        df['NAV_DATE']=pd.to_datetime(df['PRICING_DATE'], format='%d.%m.%Y')
        #iterate through isin and collect nav and merge ina blank dataframe
        isin = self.isins[0]
        nav = df.query('ISIN_CODE == @isin')[['NAV_DATE','NAV']].copy()
        if len(isins) > 1:
            for i in range(1,len(isins)):
                isin = isins[i]
                temp = df.query('ISIN_CODE == @isin')[['NAV_DATE','NAV']].copy()
                nav = nav.merge(temp, left_on='NAV_DATE', right_on='NAV_DATE')
        nav = nav.drop_duplicates()
        cols = ['index']
        for isin in isins:
            cols.append(isin)
        nav.columns = cols
        since = self.since
        return nav.query('index >= @since').sort_values('index')
    
    def create_portfolio(self, prices):
        if prices == None:
            return False
        print(f'Get prices and computes returns. prices.shape {prices.shape}')
        returns = prices.to_returns()
        
        return returns











