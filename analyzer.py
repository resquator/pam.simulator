import pandas as pd
import numpy as np

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
