from flask import Flask, render_template, request
import numpy as np
import pandas as pd

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
            isins.append([request.form.get(v),int(request.form.get(w)),np.round(perf,2)])
            listisins.append(request.form.get(v))

    ptf = pd.DataFrame(isins, columns=['ISIN','WEIGHT','PERF'], index=None)
    b100 = ptf.WEIGHT.sum()
    
    for idx, r in ptf.iterrows():
        invest = r.WEIGHT / b100 * 1000
        ptf.loc[idx,'WEIGHT'] = r.WEIGHT / b100 #int(r.WEIGHT / b100)
        ptf.loc[idx,'INVEST'] = invest
        ptf.loc[idx,'RETURN'] = (invest* (1 + (r.PERF/100))) - invest

    
    table = ptf.to_html(index=False, justify='center')

    fund = Isin(isins[0],since='01.01.2022')
    navs = fund.get_navs()
    nav_table =navs[['NAV_DATE','NAV']].sort_values('NAV_DATE', ascending=False).head(10).to_html(index=False)

    since = request.form.get('from_date')
    pf = Portfolio(listisins, since)
    s = pf.create_series()
    s_html = s.to_html(index=False)

    gl = np.round(1000 + (ptf.RETURN.sum()),2)
    html = f'<div><h5 />Portfolio Allocation since {since_when}:{table}<br>The 1000 Euro invested returns {gl} Euro</div><br><div>{nav_table}</div><div>{s.shape}</div><div>{s_html}</div>'

    return render_template('home.html', prediction_text = html)

if __name__ == "__main__":
    app.run(debug=True)
    
    