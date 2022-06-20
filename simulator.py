from fileinput import filename
from flask import Flask, render_template, request
from matplotlib.cbook import print_cycles
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from analyzer import Isin
from analyzer import Portfolio
import shutil
import ffn
import os
import glob


app = Flask(__name__, template_folder = 'templates')

#model = joblib.load("models/multi-regressor-score.pkl")

@app.route("/")
def home():


    return render_template("home.html")

@app.route('/', methods = ['POST'])
def predict():
    


    since_when = request.form.get("from_date")
    investment = request.form.get('amount')
    print(f'investment is {investment} - {type(investment)}')
    investment = int(investment)
    # iterate through isin01 to isin05 to get portfolio details
    isins= []   
    listisins = [] 
    for i in range(1,8):
        v=f'isin0{i}'
        w=f'w0{i}'        
        fund = request.form.get(v)
        fund = str(fund)
        if 'LU' in fund:
            # get fund relative performance
            fund = Isin(fund,since=since_when)
            perf = fund.get_relative_perf()
            isins.append([request.form.get(v),int(request.form.get(w)),np.round(perf,2),since_when])
            listisins.append(request.form.get(v))

    ptf = pd.DataFrame(isins, columns=['ISIN','WEIGHT','PERF','SINCE'], index=None)
    b100 = ptf.WEIGHT.sum()
    weights = []
    for idx, r in ptf.iterrows():
        invest = r.WEIGHT / b100 * investment
        ptf.loc[idx,'WEIGHT'] = r.WEIGHT / b100 #int(r.WEIGHT / b100)
        ptf.loc[idx,'INVEST'] = invest
        ptf.loc[idx,'RETURN'] = (invest* (1 + (r.PERF/100))) - invest
        weights.append(r.WEIGHT / b100)

    print(b100)
    table = ptf.to_html(index=False, justify='center', bold_rows=False, col_space=100)

    fund = Isin(isins[0],since=since_when)
    navs = fund.get_navs()
    nav_table =navs[['NAV_DATE','NAV']].sort_values('NAV_DATE', ascending=False).head(10).to_html(index=False)

    since = request.form.get('from_date')
    pf = Portfolio(listisins, since)
    s = pf.create_series()
    s_html = s.to_html(index=True)

    from datetime import datetime

#    Getting the current date and time
    dt = datetime.now()

    # getting the timestamp
    ts = datetime.timestamp(dt)
    date_time = datetime.fromtimestamp(ts)
    str_date_time = date_time.strftime("%d%m%Y%H%M%S")


    s.to_csv(f'portfolios/ptf_{str_date_time}.csv', index=False)


    """
    Call ffn() to compute the portfolio.
    """


    filename = f'portfolios/ptf_{str_date_time}.csv'
    print(filename)
    try:
        # compute the files header for the file
        file_isin_header = ','.join(listisins)
        #prices = ffn.get(f'{listisins[0]},{listisins[1]},{listisins[2]}', provider=ffn.data.csv, path=filename)
        prices = ffn.get(f'{file_isin_header}', provider=ffn.data.csv, path=filename)
        print(f'{prices.shape}')
        stats = prices.calc_stats()
        print('stats computed')
        stats.to_csv(sep=',',path=f'analysis/stats_{str_date_time}.csv')
        p = prices.to_html()
        print('prices to html')
        analysis = pd.read_csv(f'analysis/stats_{str_date_time}.csv', delimiter=',', error_bad_lines=False)
        a = analysis.to_html(bold_rows=False, justify='center', col_space=100)
        gl = np.round(investment + (ptf.RETURN.sum()),2)
        print(f'gl = {gl}')
        # max draw down
        stats.prices.to_drawdown_series().plot(figsize=(12,5))
        plt.savefig(f'static/drwadown_{str_date_time}.png', bbox_inches='tight')

        # returns histograms
        returns = prices.to_returns().dropna()
        returns.hist(figsize=(12, 5))
        plt.savefig(f'static/rethisto_{str_date_time}.png', bbox_inches='tight')

        # heatmap
        returns.plot_corr_heatmap()
        plt.savefig(f'static/heatmap_{str_date_time}.png', bbox_inches='tight')
    
        # plot perf
        prices.rebase().plot(figsize=(12,5))
        plt.savefig(f'static/perfor_{str_date_time}.png', bbox_inches='tight')
        
        
        """
        This section creates a based 100 series on the portfolio position respecting weights

        Returns:
            _type_: _description_
        """
        print(f'Entering in create_portfolio method since {since_when}')
        returns = prices.to_returns()
        # portfolios weights
        returns_1 = returns + 1
        first_index = returns_1.index[0]

        #returns_1.loc[first_index,'p1']=weights[0]
        #returns_1.loc[first_index,'p2']=weights[1]
        #returns_1.loc[first_index,'p3']=weights[2]

        for i in range(0,len(weights)):
            v = f'p{i+1}'
            returns_1.loc[first_index,v]=weights[i]


        #print(f'From {returns_1.index.min()} To {returns_1.index.max()}')
        interval = f'From {returns_1.index.min().strftime("%d %B, %Y")} To {returns_1.index.max().strftime("%d %B, %Y")}'

        last_row = returns_1.shape[0]
        """
        for i in range(1,last_row):
            returns_1.iloc[i,3] = returns_1.iloc[i-1,3] * returns_1.iloc[i,0]
            returns_1.iloc[i,4] = returns_1.iloc[i-1,4] * returns_1.iloc[i,1]
            returns_1.iloc[i,5] = returns_1.iloc[i-1,5] * returns_1.iloc[i,2]
        """
        #print('dynamic size of portfolios')
        for i in range(1,last_row):
            l = len(listisins)
            for j in range(0,l):
                returns_1.iloc[i,l+j] = returns_1.iloc[i-1,l+j] * returns_1.iloc[i,j]


        #print(f'returns_1 shape = {returns_1.shape}')
        #print(returns_1.head(4))
        
        def compute_base100(row):
            p=0
            for i in range(0,len(listisins)):
                p=p+row[f'p{i+1}']
            return p

        #returns_1['p']=returns_1['p1']+returns_1['p3']+returns_1['p2']    
        returns_1['p'] = returns_1.apply (lambda row: compute_base100(row), axis=1)

        portfolio = returns_1['p']
        file_base100 = f'portfolios/ptf_sim_{str_date_time}.csv'
        pd.DataFrame(portfolio).to_csv(file_base100)
        
        isins = ['p']
        filename = file_base100
        ptf_prices = ffn.get(f'{isins[0]}', provider=ffn.data.csv, path=filename)
        
        #resample on weekly basis
        ptf_prices = ptf_prices.resample('D').interpolate()[::7]
        print(f'weekly prices resample {ptf_prices.shape}')
        
        ptf_stats = ptf_prices.calc_stats()
        ptf_stats.to_csv(sep=',',path=f'analysis/sim_{str_date_time}.csv')
        analysis = pd.read_csv(f'analysis/sim_{str_date_time}.csv', delimiter=',', error_bad_lines=False)
        print(f'{analysis.columns.tolist()}')
        a_sim = analysis.to_html()
        
        # force the columns we wants for the table
        cols_table = ['YTD','MTD','1m','3m','6m','1Y','Total Return']
        cols_greek = ['Daily Sharpe','Daily Vol (ann.)','Max Drawdown']
        pd_cols = pd.DataFrame(cols_table + cols_greek, columns=['labels'])
        print(analysis.head(5))
        pd_cols=pd_cols.merge(analysis, left_on='labels', right_on='Stat', how='left').fillna('-')
        a_sim_summary = pd_cols[['labels','p']].to_html(index=False, col_space=100, justify='center')

        # plot perf_port
        ptf_prices.rebase().plot(figsize=(12,5))
        plt.savefig(f'static/perfor_ptf_{str_date_time}.png', bbox_inches='tight')
        

        html = f'<div class="naija-flag"><h3>Portfolio simulation {interval}</h3></div><br>The {investment} Euro invested returns {gl} Euro. This means a {np.round(((gl/investment)-1)*100,2)}% performance.<hr>{table}<hr>{a_sim_summary}<hr><img src="/static/perfor_ptf_{str_date_time}.png"><hr>{a_sim}<br><div><table><tr /><td /><h2>Portfolio details</h2><img src="/static/heatmap_{str_date_time}.png"><td /><img src="/static/rethisto_{str_date_time}.png"></table></div><div><table valign="top"><tr /><td />{a}<td /><img src="/static/perfor_{str_date_time}.png"></table></div></div>'
        return render_template('home.html', prediction_text = html)

    except OSError as err:
        print("OS error: {0}".format(err))
        return render_template('home.html', prediction_text = f'Error {err}')

    except ValueError:
        print("Could not convert data to an integer.")
        return render_template('home.html', prediction_text = f'Error dtypes')

    except BaseException as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return render_template('home.html', prediction_text = f'Error {err}')

        raise    

if __name__ == "__main__":
    app.run(debug=True)
    
    