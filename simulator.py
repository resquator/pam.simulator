from fileinput import filename
from flask import Flask, render_template, request
from matplotlib.cbook import print_cycles
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from analyzer import Isin
from analyzer import Portfolio

import ffn



app = Flask(__name__, template_folder = 'templates')

#model = joblib.load("models/multi-regressor-score.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/', methods = ['POST'])
def predict():
    


    since_when = request.form.get("from_date")
    investment = request.form.get('amount')
    investment = 100000
    # iterate through isin01 to isin05 to get portfolio details
    isins= []   
    listisins = [] 
    for i in range(1,6):
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
    
    for idx, r in ptf.iterrows():
        invest = r.WEIGHT / b100 * investment
        ptf.loc[idx,'WEIGHT'] = r.WEIGHT / b100 #int(r.WEIGHT / b100)
        ptf.loc[idx,'INVEST'] = invest
        ptf.loc[idx,'RETURN'] = (invest* (1 + (r.PERF/100))) - invest

    
    table = ptf.to_html(index=False, justify='center')

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
    try:
        prices = ffn.get(f'{listisins[0]},{listisins[1]},{listisins[2]}', provider=ffn.data.csv, path=filename)

        stats = prices.calc_stats()
        stats.to_csv(sep=',',path=f'analysis/stats_{str_date_time}.csv')
        p = prices.to_html()
        analysis = pd.read_csv(f'analysis/stats_{str_date_time}.csv', delimiter=',', error_bad_lines=False)
        a = analysis.to_html()
        gl = np.round(investment + (ptf.RETURN.sum()),2)

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

        html = f'<div class="naija-flag"><h5 />Portfolio Allocation since {since_when}:{table}<br>The {investment} Euro invested returns {gl} Euro. This means a {np.round(((gl/investment)-1)*100,2)}% performance.<br><div><table><tr /><td /><img src="/static/heatmap_{str_date_time}.png"><td /><img src="/static/rethisto_{str_date_time}.png"></table></div><div><table valign="top"><tr /><td />{a}<td /><img src="/static/perfor_{str_date_time}.png"></table></div></div>'
    except:
        html = 'No answer'
    
    return render_template('home.html', prediction_text = html)

if __name__ == "__main__":
    app.run(debug=True)
    
    