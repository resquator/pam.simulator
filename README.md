# pam.portfolio-simulator

## This repository contains the POC for a new portfolio simulator.

A portfolio simulator helps to simulate the behaviour of a portfolio created on the top of PAM Funds.

The simulation compute abase 100 weighted portfolio with funds selected by the user and weight assigned.
The simulation is sent to a temporay file which will be loaded in the ffn() library to get prices series and compute stats().

The project works with
- flask
- numpy
- pandas
- ffn



To run the project, be sure that flask is installed on the computer
- you can run: pip install flask --upgrade (if error)
- python simulator.py (to start the project)
- cmd+click on the ip:127.0.0.1:5000 (to display the main page)

