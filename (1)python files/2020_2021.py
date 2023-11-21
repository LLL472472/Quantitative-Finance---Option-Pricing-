# -*- coding: utf-8 -*-
"""MIE1622_A2_2020_2021.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kNhBDrB3wEdIOu1C25ps_AYlo7dmviS9
"""

# MIE 1622
# Assignment 2
# 2020, 2021
# Jiacheng Li
# ID: 1005138405



# Import libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
!pip install cplex
import cplex

!brew install git
!brew install pcre2
!brew cleanup
!brew install pkg-config
!brew install gcc
!sudo apt-get install coinor-libipopt-dev
!conda activate </Users/chelseali/opt/anaconda3>
!pip install cyipopt
import cyipopt as ipopt

# Complete the following functions


# Strategy 1
# 'Buy and hold'
# hold initial portfolio for the entire investment horizon of 2 years
def strat_buy_and_hold(x_init, cash_init, mu, Q, cur_prices):
   x_optimal = x_init
   cash_optimal = cash_init
   return x_optimal, cash_optimal


# Strategy 2
# 'Equally weighted': (1/n) protfolio strategy, where n is the number of assets
# re-balance the portfolio in each period 
# as the number of shares changes, even when Wi = 1/n stays the same in each period
def strat_equally_weighted(x_init, cash_init, mu, Q, cur_prices):
    # Calculate the total portfolio value
    Total_Portfolio_Value = np.dot(cur_prices, x_init) + cash_init
    # There are n assets
    n = len(x_init)
    # Equal weight for the n stocks, equal weight = 1/n
    w = np.ones((n)) / n
    # Calculate each asset value, total value * each weight
    each_asset_value = w * Total_Portfolio_Value
    # Rounding procedure of number of shares  
    x_optimal = np.floor(each_asset_value / cur_prices)
    # Calculate the transcation cost
    # The variable fee is due to the difference between the selling and bidding price of a stock
    # and 0.5% of the traded volume
    transaction_cost = 0.005 * np.dot(cur_prices,abs(x_optimal-x_init))
    # Calculate the cash optimal based on x_optimal
    cash_optimal = Total_Portfolio_Value - np.dot(cur_prices,x_optimal) - transaction_cost   
    return x_optimal, cash_optimal


# Strategy 3
# 'Minimum variance'
# hold initial portfolio for the entire investment horizon of 2 years
def strat_min_variance(x_init, cash_init, mu, Q, cur_prices):
    # Calculate the total portfolio value
    Total_Portfolio_Value = np.dot(cur_prices, x_init) + cash_init
    # There are n assets
    n = len(x_init)
    # Initialize Cplex object
    cpx = cplex.Cplex()
    # Miximize it, min variance
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    # Set c,lb,ub,A to cplex slover 
    c = np.zeros((n))
    lb = np.zeros((n))
    ub = np.ones((n))
    A = []
    for i in range(n):
        A.append([[0,1],[1,0]]) 
    var_names = ['w_%s'% i for i in range(1, n+1)]
    #add some variables to cplex model
    cpx.linear_constraints.add(rhs=[1.0,0],senses='EG')   
    cpx.variables.add(obj = c,lb = lb,ub = ub,columns = A,names = var_names)
    # Add quadratic part of objective function
    Qmat = [[list(range(n)),list(2*Q[j,:])] for j in range(n)] 
    # Quadratic objective
    cpx.objective.set_quadratic(Qmat)
    cpx.parameters.threads.set(6)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()
    # calculate the optimal weights by cpx 
    w = np.array(cpx.solution.get_values()) 
    # Calculate each asset value, total value * each weight
    each_asset_value = w * Total_Portfolio_Value
    # Rounding procedure of number of shares  
    x_optimal = np.floor(each_asset_value / cur_prices)
    # Calculate the transcation cost
    # The variable fee is due to the difference between the selling and bidding price of a stock
    # and 0.5% of the traded volume
    transaction_cost = 0.005 * np.dot(cur_prices,abs(x_optimal-x_init))
    # Calculate the cash optimal based on x_optimal
    cash_optimal = Total_Portfolio_Value - np.dot(cur_prices,x_optimal) - transaction_cost  
    return x_optimal, cash_optimal


# Strategy 4
# 'Maximum Sharp ratio'
# hold initial portfolio for the entire investment horizon of 2 years
def strat_max_Sharpe(x_init, cash_init, mu, Q, cur_prices):
    # Calculate the total portfolio value
    Total_Portfolio_Value = np.dot(cur_prices, x_init) + cash_init
    n = len(x_init) + 1
    # Calculate the daily risk free rate 
    daily_rf = 1.025**(1.0/252) - 1
    # Calculate the rate difference
    rate_diff = mu - daily_rf 
    # Initialize Cplex object
    cpx = cplex.Cplex()
    # Miximize it, m(- sharp ratio)
    cpx.objective.set_sense(cpx.objective.sense.minimize)   
    # Set c,lb,ub to cplex slover 
    c = [0.0]*n 
    lb = [0.0]*n
    ub = [np.inf] * n  
    A = []
    # Add new column and row for the risk-free asset
    Q_new = np.append(Q, np.zeros(((n-1),1)),axis=1)
    Q_new = np.vstack([Q_new,np.zeros((n))])  
    for i in range(n-1):
        A.append([[0,1],[rate_diff[i],1.0]])
    A.append([[0,1],[0,-1.0]])
    var_names = ["y_%s" % i for i in range(1,n+1)]
    # Add some variables to cplex model
    cpx.linear_constraints.add(rhs=[1.0, 0], senses="EE")
    cpx.variables.add(obj=c, lb=lb, ub = ub, 
                      columns=A, names = var_names)
    # Add quadratic part of objective function
    Qmat =[[list(range(n)),list(2*Q_new[j,:])] for j in range(n)]
    # Quadratic objective
    cpx.objective.set_quadratic(Qmat)
    #set parameters
    cpx.parameters.threads.set(6)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()
    if cpx.solution.get_status_string()== 'infeasible':
      x_optimal = x_init
      cash_optimal = cash_init
      w1 = ( x_init * cur_prices ) / Total_Portfolio_Value
    else:
      # calculate the optimal weights by cpx 
      w = np.array(cpx.solution.get_values())
      w = w[0:(n-1)]/w[(n-1)]
      # Calculate each asset value, total value * each weight
      each_asset_value = w * Total_Portfolio_Value
      # Rounding procedure of number of shares  
      x_optimal = np.floor(each_asset_value / cur_prices)
      # Calculate the transcation cost
      # The variable fee is due to the difference between the selling and bidding price of a stock
      # and 0.5% of the traded volume
      transaction_cost = 0.005 * np.dot(cur_prices,abs(x_optimal-x_init))
      # Calculate the cash optimal based on x_optimal
      cash_optimal = Total_Portfolio_Value - np.dot(cur_prices,x_optimal) - transaction_cost
    return x_optimal, cash_optimal

# Strategy 5
# 'Equal risk contributions'
# compute a portfolio that has equal risk contributions to std for each period and re-balance accordingly
# compute the gradient of the objective function
def strat_equal_risk_contr(x_init, cash_init, mu, Q, cur_prices):
    Total_Portfolio_Value = np.dot(cur_prices, x_init) + cash_init 
    n = len(x_init)
    class erc(object):
        def __init__(self):
            pass
        def objective(self, x):
            y = x * np.dot(Q, x) # The callback for calculating the objective
            fval = 0
            for i in range(n):
                for j in range(i,n):
                    xij = y[i] - y[j]
                    fval = fval + xij*xij
            fval = 2*fval
            return fval
        def gradient(self, x):
            grad = np.zeros(n) # The callback for calculating the gradient
            y = x * np.dot(Q, x) # Insert your gradient computations here
            for i in range(n):
                for j in range(n):
                    diff1 = np.dot(Q[i],x) + np.dot(Q[i][i],x[i])
                    diff2 = np.dot(Q[i][j], x[i])
                    delta_g = (y[i]-y[j]) * (diff1 - diff2)
                    grad[i] = grad[i] + delta_g
                grad[i] = 2 * 2 * grad[i]
            return grad
        def constraints(self, x):
            return [1.0] * n # The callback for calculating the constraints
        def jacobian(self, x):
            return np.array([[1.0] * n]) # The callback for calculating the Jacobian
    w_init = (x_init * cur_prices) / Total_Portfolio_Value # initial weight distribution
    lb = [0.0] * n  # lower bounds on variables
    ub = [1.0] * n  # upper bounds on variables
    cl = [1]        # lower bounds on constraints
    cu = [1]        # upper bounds on constraints 
    nlp = ipopt.Problem(n=len(w_init), m=len(cl), problem_obj=erc(), lb=lb, ub=ub, cl=cl, cu=cu) # Define IPOPT problem
    nlp.add_option('jac_c_constant'.encode('utf-8'), 'yes'.encode('utf-8')) # Set the IPOPT options
    nlp.add_option('hessian_approximation'.encode('utf-8'), 'limited-memory'.encode('utf-8'))
    nlp.add_option('mu_strategy'.encode('utf-8'), 'adaptive'.encode('utf-8'))
    nlp.add_option('tol'.encode('utf-8'), 1e-10)
    w_optimal, info = nlp.solve(w_init) 
    each_asset_value = w_optimal * Total_Portfolio_Value # Calculate each asset value, total value * each weight
    x_optimal = np.floor(each_asset_value / cur_prices) # Rounding procedure of number of shares    
    # Calculate the transcation cost
    # The variable fee is due to the difference between the selling and bidding price of a stock
    # and 0.5% of the traded volume                                 
    transaction_cost = 0.005 * np.dot(cur_prices,abs(x_optimal-x_init))             
    cash_optimal = Total_Portfolio_Value - np.dot(cur_prices,x_optimal) - transaction_cost # Calculate the cash optimal based on x_optimal
    return x_optimal, cash_optimal

# Strategy 6
# 'Leveraged equal risk contributions'
# take long 200% position in equal risk contributions portfolio and short risk-free asset for each period
def strat_lever_equal_risk_contr(x_init, cash_init, mu, Q, cur_prices):
    Total_Portfolio_Value = np.dot(cur_prices, x_init) + cash_init 
    n = len(x_init)
    #Use IPOPT solver to calculate equal risk contribution
    class erc(object):
        def __init__(self):
            pass
        def objective(self, x):
            y = x * np.dot(Q, x) # The callback for calculating the objective
            fval = 0
            for i in range(n):
                for j in range(n):
                    xij = y[i] - y[j]
                    fval = fval + xij*xij
            fval = 2*fval
            return fval
        def gradient(self, x):
            grad = np.zeros(n) # The callback for calculating the gradient
            y = x * np.dot(Q, x)
            for i in range(n):
                for j in range(n):
                    diff1 = np.dot(Q[i],x) + np.dot(Q[i][i],x[i])
                    diff2 = np.dot(Q[i][j], x[i])
                    delta_g = (y[i]-y[j]) * (diff1 - diff2)
                    grad[i] = grad[i] + delta_g
                grad[i] = 2 * 2 * grad[i] 
            return grad
        def constraints(self, x):
            return [1.0] * n # The callback for calculating the constraints
        def jacobian(self, x):
            return np.array([[1.0] * n]) # The callback for calculating the Jacobian
    borrow = init_value # shorting amount 
    r_rf = 0.025 #risk-free rate
    interest = borrow * (r_rf / 6)
    if period == 1: # take long 200% position in first period
        Total_Portfolio_Value = Total_Portfolio_Value + borrow
    lb = [0.0] * n  # lower bounds on variables
    ub = [1.0] * n  # upper bounds on variables
    cl = [1]        # lower bounds on constraints
    cu = [1]        # upper bounds on constraints
    w_init = (x_init * cur_prices) / Total_Portfolio_Value # initial weight distribution
    nlp = ipopt.Problem(n=len(w_init), m=len(cl), problem_obj=erc(), lb=lb, ub=ub, cl=cl, cu=cu) # Define IPOPT problem
    nlp.add_option('jac_c_constant'.encode('utf-8'), 'yes'.encode('utf-8')) # Set the IPOPT options
    nlp.add_option('hessian_approximation'.encode('utf-8'), 'limited-memory'.encode('utf-8'))
    nlp.add_option('mu_strategy'.encode('utf-8'), 'adaptive'.encode('utf-8'))
    nlp.add_option('tol'.encode('utf-8'), 1e-10)
    w_lerc, info = nlp.solve(w_init)
    w_lerc = np.asarray(w_lerc)
    w_optimal = w_lerc*(1/w_lerc.sum())
    each_asset_value = w_optimal * Total_Portfolio_Value # Calculate each asset value
    x_optimal = np.floor(each_asset_value / cur_prices) # Rounding procedure of number of shares  
    # Calculate the transcation cost
    # The variable fee is due to the difference between the selling and bidding price of a stock
    # and 0.5% of the traded volume                                 
    transaction_cost = 0.005 * np.dot(cur_prices,abs(x_optimal-x_init))             
    cash_optimal = Total_Portfolio_Value - np.dot(cur_prices,x_optimal) - transaction_cost - interest # Calculate the cash optimal based on x_optimal
    return x_optimal, cash_optimal

# Strategy 7
# 'Robust mean-variance optimization'
# compute a robust mean-variance portfolio for each period and re-balance
# select target risk estimation error and target return
def strat_robust_optim(x_init, cash_init, mu, Q, cur_prices):
    Total_Portfolio_Value = np.dot(cur_prices, x_init) + cash_init 
    n = len(x_init)
    w_init = [1 / n] * n # Initial weight distribution for 1/n portfolio

    r_rf = 0.025 #risk-free rate
    daily_rf = r_rf / 252
    Portf_Retn = daily_rf 

    cpx = cplex.Cplex() # Initialize Cplex object
    cpx.objective.set_sense(cpx.objective.sense.minimize) # do minimize

    c  = [0.0] * n
    lb = [0.0] * n # lower bounds on variables
    ub = [1.0] * n # upper bounds on variables
     
    var_matr = np.diag(np.diag(Q)) 
    rob_init = np.dot(w_init, np.dot(var_matr, w_init)) # return estimation error 
    rob_bnd  = rob_init 
                                    
    A = []
    for k in range(n):
        A.append([[0,1],[1.0,mu[k]]])
    var_names = ["w_%s" % i for i in range(1,n+1)]
    cpx.linear_constraints.add(rhs=[1.0,Portf_Retn], senses="EG")
    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    Qmat = [[list(range(n)), list(2*Q[k,:])] for k in range(n)]
    cpx.objective.set_quadratic(Qmat)
    Qcon = cplex.SparseTriple(ind1=var_names, ind2=range(n), val=np.diag(var_matr))
    cpx.quadratic_constraints.add(rhs=rob_bnd, quad_expr=Qcon, name="Qc")
    cpx.parameters.threads.set(4)
    cpx.parameters.timelimit.set(60)
    cpx.parameters.barrier.qcpconvergetol.set(1e-12)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()

    if cpx.solution.get_status_string()== 'infeasible':
      x_optimal = x_init
      cash_optimal = cash_init
      w1 = ( x_init * cur_prices ) / Total_Portfolio_Value

    else:
      # calculate the optimal weights by cpx 
      w = np.array(cpx.solution.get_values())
      # Round near-zero portfolio weights
      w[w<1e-6] = 0
      w_optimal = w / np.sum(w)
      # Calculate each asset value, total value * each weight
      each_asset_value = w_optimal * Total_Portfolio_Value
      # Rounding procedure of number of shares  
      x_optimal = np.floor(each_asset_value / cur_prices)
      # Calculate the transcation cost
      # The variable fee is due to the difference between the selling and bidding price of a stock
      # and 0.5% of the traded volume
      transaction_cost = 0.005 * np.dot(cur_prices,abs(x_optimal-x_init))
      # Calculate the cash optimal based on x_optimal
      cash_optimal = Total_Portfolio_Value - np.dot(cur_prices,x_optimal) - transaction_cost
    return x_optimal, cash_optimal

# 2. Analyze my results
# Input file
input_file_prices = 'Daily_closing_prices.csv'

# Read data into a dataframe
df = pd.read_csv(input_file_prices)

# Convert dates into array [year month day]
def convert_date_to_array(datestr):
    temp = [int(x) for x in datestr.split('/')]
    return [temp[-1], temp[0], temp[1]]

dates_array = np.array(list(df['Date'].apply(convert_date_to_array)))
data_prices = df.iloc[:, 1:].to_numpy()
dates = np.array(df['Date'])



# Find the number of trading days in Nov-Dec 2019 and
# compute expected return and covariance matrix for period 1
day_ind_start0 = 0
day_ind_end0 = len(np.where(dates_array[:,0]==2019)[0])
# Calculate the daily reuturn rate
cur_returns0 = data_prices[day_ind_start0+1:day_ind_end0,:] / data_prices[day_ind_start0:day_ind_end0-1,:] - 1
# During period 1, the avg daily return rates of 20 
mu = np.mean(cur_returns0, axis = 0)
Q = np.cov(cur_returns0.T)

# Remove datapoints for year 2019
# At the begining of 2020
data_prices = data_prices[day_ind_end0:,:]
dates_array = dates_array[day_ind_end0:,:]
dates = dates[day_ind_end0:]



# Initial positions in the portfolio
# HOG = 902 shares
# VZ = 17500 shares
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17500])
# Initial value of the portfolio
init_value = np.dot(data_prices[0,:], init_positions)
print('\nInitial portfolio value = $ {}\n'.format(round(init_value, 2)))
# Initial portfolio weights at the first day in period 2
w_init = (data_prices[0,:] * init_positions) / init_value

# Number of periods, assets, trading days starting from 2020
N_periods = 6*len(np.unique(dates_array[:,0])) # 6 periods per year
N = len(df.columns)-1
N_days = len(dates)

# Annual risk-free rate for years 2020-2021 is 2.5%
r_rf = 0.025
# Annual risk-free rate for years 2008-2009 is 4.5%
r_rf2008_2009 = 0.045
# Annual risk-free rate for year 2022 is 3.75%
r_rf2022 = 0.0375


# Number of strategies
strategy_functions = ['strat_buy_and_hold', 'strat_equally_weighted', 'strat_min_variance', 'strat_max_Sharpe','strat_equal_risk_contr','strat_lever_equal_risk_contr','strat_robust_optim']
strategy_names     = ['Buy and Hold', 'Equally Weighted Portfolio', 'Minimum Variance Portfolio', 'Maximum Sharpe Ratio Portfolio','Equal risk contributions','Leveraged equal risk contributions','Robust mean-variance optimization']

N_strat = len(strategy_functions)  # uncomment this in your code
fh_array = [strat_buy_and_hold, strat_equally_weighted, strat_min_variance, strat_max_Sharpe, strat_equal_risk_contr, strat_lever_equal_risk_contr, strat_robust_optim]

portf_value = [0] * N_strat
x = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
cash = np.zeros((N_strat, N_periods),  dtype=np.ndarray)

# Analyze my results:
# Outputs for 12 periods (years 2020 and 2021)

print('\nInitial portfolio value = $ {}\n'.format(round(init_value, 2)))

for period in range(1, N_periods+1):
   # Compute current year and month, first and last day of the period
   if dates_array[0, 0] == 20:
       cur_year  = 20 + math.floor(period/7)
   else:
       cur_year  = 2020 + math.floor(period/7)

   cur_month = 2*((period-1)%6) + 1
   day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
   day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])
   print('\nPeriod {0}: start date {1}, end date {2}'.format(period, dates[day_ind_start], dates[day_ind_end]))
   # Prices for the current day
   cur_prices = data_prices[day_ind_start,:]
   # Execute portfolio selection strategies
   for strategy  in range(N_strat):

      # Get current portfolio positions
      if period == 1:
         curr_positions = init_positions
         curr_cash = 0
         portf_value[strategy] = np.zeros((N_days, 1))
      else:
         curr_positions = x[strategy, period-2]
         curr_cash = cash[strategy, period-2]
         
      # Compute strategy
      x[strategy, period-1], cash[strategy, period-1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
      # Verify that strategy is feasible (you have enough budget to re-balance portfolio)
      # Check that cash account is >= 0
      # Check that we can buy new portfolio subject to transaction costs

      if cash[strategy, period-1] < 0:
          # leveraged equal risk contributions strategy
          if cash[5,0]<0:
              portfolio_V = (np.dot(cur_prices,curr_positions) + curr_cash) * 2
          else:
              portfolio_V = np.dot(cur_prices,curr_positions) + curr_cash
                
          # Change the portfolio by changing the ratio of each stock
          # cur_total_portfolio_value = np.dot(cur_prices,curr_positions) + curr_cash
          ratio = x[strategy, period-1]/np.sum(x[strategy, period-1])
          excess_cash = abs(cash[strategy, period-1])*ratio
          excess_position = np.ceil(excess_cash/cur_prices)
          # Calculate the new optimal protfolio
          x[strategy, period-1] = x[strategy, period-1] - excess_position
          # Calculate the new transaction cost
          new_transaction_cost = np.dot(cur_prices , abs(x[strategy, period-1]-curr_positions)) * 0.005
          # Calculate the new cash account value
          cash[strategy, period-1] = portfolio_V - np.dot(cur_prices,x[strategy, period-1]) - new_transaction_cost
            
      # Compute portfolio value
      p_values = np.dot(data_prices[day_ind_start:day_ind_end+1,:], x[strategy, period-1]) + cash[strategy, period-1]
      #if using leveraged equal risk contributions strategy
      if strategy ==5: 
            portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1)) - init_value
      else:
            portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1))
      print('  Strategy "{0}", value begin = $ {1:.2f}, value end = $ {2:.2f}'.format( strategy_names[strategy], 
             portf_value[strategy][day_ind_start][0], portf_value[strategy][day_ind_end][0]))
      
   # Compute expected returns and covariances for the next period
   cur_returns = data_prices[day_ind_start+1:day_ind_end+1,:] / data_prices[day_ind_start:day_ind_end,:] - 1
   mu = np.mean(cur_returns, axis = 0)
   Q = np.cov(cur_returns.T)

# Plot results
###################### Insert your code here ############################


# Daily value of 7 portfolio strategies
plt.figure(figsize=(20,10))
plt.plot(portf_value[0],label='Buy and Hold')
plt.plot(portf_value[1],label='Equally Weighted Portfolio')
plt.plot(portf_value[2],label='Minimum Variance Portfolio')
plt.plot(portf_value[3],label='Maximum Sharpe Ratio Portfolio')
plt.plot(portf_value[4],label='Equal risk contributions')
plt.plot(portf_value[5],label='Leveraged equal risk contributions')
plt.plot(portf_value[6],label='Robust mean-variance optimization')

plt.legend()
plt.title('Figure 1 (2020-2021): Daily Portfolio Values for 7 Strategies', fontsize=20)
plt.xlabel('Date: # of days', fontsize=20)
plt.ylabel('The Total Portfolio Values', fontsize=20)
plt.savefig('Figure 1 (2020-2021): Daily Portfolio Values for 7 Strategies.png')
plt.show()

# We can see that before the first 40 days overall, strategy "Buy and Hold" underperformed compared to the other strategies with the  lowest returns. 

# After that, we find that "Leveraged equal risk contributions" strategy struggled in the first half of 2021. But it performances well after the first 6 months.

# Strategies "Buy and Hold" and "Minimum Variance Portfolio" showed consistent performance throughout the whole period compared to others,

# while strategy 4 "Maximum Sharpe Ratio Portfolio" showed the highest returns overall, with a significant increase in value in the first half of 2020.

# In summary, strategies "Maximum Sharpe Ratio Portfolio" and "Leveraged equal risk contributions" showed the highest returns in year of 2021. 

# But they showed mixed results over the period.

#Plot one chart in Python that illustrates maximum drawdown of your portfolio for each of the 12 periods
dfValues = pd.DataFrame()
for i in range(7):
    dfValues[str(i+1)] = portf_value[i].flatten()
dfList = []
indexArray = [[0,40],[41,82],[83,124],[125,167],[168,211],[212,252],[253,291],[292,334],[335,376],[377,419],[420,462],[463,504]]
for i in range(12):
    dfList.append(dfValues.loc[indexArray[i][0]:indexArray[i][1]])
maxDD_array = []
for i in range(12):
    periodArray = []
    for j in range(7):
        dfBuffer = ((dfList[i][str(j+1)] - dfList[i][str(j+1)].expanding().max()) / dfList[i][str(j+1)].expanding().max())*100
        dfBuffer = dfBuffer.abs()
        bufferVal = dfBuffer.max()
        periodArray.append(bufferVal)
    maxDD_array.append(periodArray)
dfMaxDD = pd.DataFrame()
for i in range(12):
    dfMaxDD[i+1] = maxDD_array[i] 
dfMaxDD = dfMaxDD.transpose()
dfMaxDD = dfMaxDD.rename(columns={0: "Buy and hold", 1: "Equally weighted", 2: "Minimum variance", 3: "Maximum Sharpe ratio", 4: "Equal risk contributions", 5: "Leveraged equal risk contributions", 6: "Robust optimization"})
dfMaxDD.plot(figsize=(20,10))
plt.title('Figure 2 (2020-2021): Maximum Drawdown of Portfolio', fontsize=20)
plt.xlabel('Period', fontsize=20)
plt.ylabel('Maximum drawdown (%)', fontsize=20)
plt.savefig('Figure 2 (2020-2021): Maximum Drawdown of Portfolio.png')
plt.show()

# Looking at the Figure 2 (2020-2021): Maximum Drawdown of Portfolio.png, 
# we can see that the strategies "Leveraged equal risk contributions" experienced the highest maximum drawdowns in 2021,
# During the first 6 periods, the difference in maximum drawdowns of portfolios is obvious. 
# After that, the 7 strategies have similar maximum drawdowns.
# "Buy and Hold" is the one which has the fewest fluctuations of maximum drawdown line, indicating more stability in its returns.

# A large maximum drawdown means that the portfolio has experienced a significant decline from its peak value.
# This can be an indication of high risk and volatility in the portfolio. 
# It is important for investors to consider the maximum drawdown when evaluating a portfolio's risk and potential returns, 
# as it can have a significant impact on the overall performance and success of the portfolio. 
# A portfolio with a large maximum drawdown may require a longer time to recover to its previous peak value,
# which can negatively impact the investor's returns and investment goals.

# Plot dynamic changes in portfolio allocations for strategy under strategy 7
stocks = df.columns[1:]
w = []
for period in range(1, N_periods+1):
    w.append(x[6, period-1]/sum(x[6, period-1]))
df7 = pd.DataFrame(np.array(w), columns=stocks, index=[1,2,3,4,5,6,7,8,9,10,11,12])
df7.plot(figsize=(20,10))
plt.title('Figure 3 (2020-2021): Dynamic Changes in portfolio allocations under strategy 7', fontsize=20)
plt.xlabel('Period', fontsize=20)
plt.ylabel('portfolio weights', fontsize=20)
plt.savefig('Figure 3 (2020-2021): Dynamic Changes in portfolio allocations under strategy 7.png')
plt.show()

# From figure 3, we can see the asset called "HOG" has the biggest weighting during the most time,
# Which is around 0.8 at the end.
# And it is the only one whose weights are larger than 0.5.
# “T” has the second largest weighting generally, whose proportion dominates in period 9.
# But its weights are smaller than 0.2 most time. “F”, “SONY”, and “AAPL” are three assets having larger weights compared with others.

# Plot dynamic changes of portfolio weights for minimize variance strategy 3
# Get the stocks names
stocks_names = df.columns[1:]
# Create a list for weights
w_minv = []
for period in range(1, N_periods+1):
    w_minv.append(x[2, period-1]/sum(x[2, period-1]))
# There are 12 periods, need index for 12
df_minv = pd.DataFrame(np.array(w_minv), columns=stocks_names, index=[1,2,3,4,5,6,7,8,9,10,11,12])
df_minv.plot(figsize=(20,10))
plt.title('Figure 4 (2020-2021): Dynamic Changes of Weights in strategy of strategy 3 Minimum Variance Portfolio', fontsize=20)
plt.xlabel('Period', fontsize=20)
plt.ylabel('portfolio weights', fontsize=20)
plt.savefig('Figure 4 (2020-2021): Dynamic Changes of Weights in strategy of strategy 3 Minimum Variance Portfolio.png')
plt.show()

# From figure 4 (2020-2021 under strategy 3), we can see the asset called “HOG” has the biggest weightings during the most time, which are above 0.5 usually.
# There are two times that the weights of “HOG” are bigger than 0.8. 
# And it is the only one whose weights are larger than 0.5. 
# “T” has the second largest weighting generally, whose proportion dominates in period 9.
# But its weights are smaller than 0.2 most time. “F”, “SONY”, and “AAPL” are three assets having larger weights compared with others.

#Plot dynamic Changes of portfolio weights for maximize Sharpe Ratio strategy 4
# Get the stocks names
# stocks_names = df.columns[1:]
# Create a list for weights
w_maxs = []
for period in range(1, N_periods+1):
    w_maxs.append(x[3, period-1]/sum(x[3, period-1]))
# There are 12 periods, need index for 12
df_maxs = pd.DataFrame(np.array(w_maxs), columns=stocks_names, index=[1,2,3,4,5,6,7,8,9,10,11,12])
df_maxs.plot(figsize=(20,10))
plt.title('Figure 5 (2020-2021): Dynamic Changes of Weights in strategy 4 Maximum Sharpe Ratio Portfolio', fontsize=20)
plt.xlabel('Period', fontsize=20)
plt.ylabel('portfolio weights', fontsize=20)
plt.savefig('Figure 5 (2020-2021): Dynamic Changes of Weights in strategy 4 Maximum Sharpe Ratio Portfolio.png')
plt.show()

# From figure 5 (2020-2021 under strategy 4), we can see there are three assets whose weights are larger than 0.8 sometimes, even almost 1.0. 
# And many assets can have weights bigger than 0.5

# Does robust portfolio seelection strategy reduce trading as compared with strategies 3 and 4.

# Yes, robust portfolio selection strategy reduced trading.
# We can see from figure 5 (2020-2021 under strategy 4), there are two assets’ weights approaching to 1.0. 
# In figure 3 (2020-2021 under strategy 7), there are two times that assets’ weights are bigger than 0.8. 
# However, we found there is only one weight of 20 assets over 12 periods, which is larger than 0.8 one time. 
# Robust portfolio selection strategies can reduce trading 
# because they typically involve a long-term investment approach 
# that emphasizes diversification and risk management over short-term market timing and frequent trading.

# Compare the last 3 strategies between each other
# And to the first 4 strategies
# Their performance relative to each other
# Which to select, why

# From figure 1 (2020-2021), we can know “Equal risk contributions” and “Robust mean-variance optimization” perform better than “Leveraged equal risk contributions” do,
# which have the lowest return during days 0 to 200. After that, the total portfolio value of “Leveraged equal risk contributions” increases a lot. 
# And we found “Equal risk contributions” and “Robust mean-variance optimization” have similar trends of portfolio values. 
# But we cannot say “Leveraged equal risk contributions” is the best among the new three strategies. 
# It has experienced a most significant decline from its peak value based on figure 2 (2020-2021).
# It is shown that before the first 40 days overall, the strategy "Buy and Hold" underperformed compared to the other strategies with the lowest returns.
# After that, we find that the "Leveraged equal risk contributions" strategy struggled in the first half of 2021. 
# But it performs well after the first 6 months. 
# Strategies "Buy and Hold" and "Minimum Variance Portfolio" showed consistent performance throughout the whole period compared to others, 
# while strategy 4 "Maximum Sharpe Ratio Portfolio" showed the highest returns overall, with a significant increase in value in the first half of 2020. 
# Also, it is shown that “Buy and Hold” and “Minimum Variance Portfolio” have similar trends and total values. 
# These two strategies do not perform better than “Equal risk contributions” or “Robust mean-variance optimization”.
# In summary, strategies "Maximum Sharpe Ratio Portfolio" and "Leveraged equal risk contributions" showed the highest returns in the year 2021. 
# "Maximum Sharpe Ratio Portfolio" is the best one with the highest value most time. But they showed mixed results over the period. 

# I would like to choose "Maximum Sharpe Ratio Portfolio" to manage my portfolio. 
# In the long term, it can maintain the biggest total values and have acceptable maximum drawdowns.