# Install the packages 
!pip install yfinance

# Import the libraries
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import seaborn as sns
import yfinance as yf

# Select the stocks to test the model
stocks = ['ORCL','EL','NOW','ADI','ISRG','CVS','T','EOG','REGN','TJX']
stocks_compl = ["AAPL"] + stocks

# Download the data for first stock and store it 
apl = yf.Ticker("AAPL")
aapl = apl.history(period ='5y')

# Calculate the returns for the first stock and store it
pct_aapl = aapl["Close"].pct_change()

prices_aapl = aapl["Close"]
returns = pct_aapl.to_frame()
prices = prices_aapl.to_frame()

# Store the generated returns and downloaded data
returns = returns.rename(columns={"Close": stocks_compl[0]})
prices = prices.rename(columns={"Close": stocks_compl[0]})

# Downloading the stocks' data using the 'yfinance' library
for stock in stocks:
    df1 = yf.Ticker(stock)
    df = df1.history(period='5y')
    df_pct = df["Close"].pct_change()
    df_price = df["Close"]
    returns = returns.join(df_pct).rename(columns={"Close": stock})
    prices = prices.join(df_price).rename(columns={"Close": stock})

# Remove the empty cells from the returns dataframe 
returns = returns.dropna()

# ************************************************************************ TREE CLUSTERING ****************************************************************************

# Compute the correlation matrix 
corr = returns.corr()

# Plot the correlation matrix
ax = sns.heatmap(corr, cmap="coolwarm")

# Calculate the disctance matrix
d_corr = np.sqrt(0.5*(1-corr))

# Compute the linkage matrix using the distance matrix 
link = linkage(d_corr, 'single')
Z = pd.DataFrame(link)

# Plot the dendogram 
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, labels = stocks_compl)
plt.show()

# *********************************************************************** QUASI-DIAGONALIZATION ***********************************************************************

# Make a function which returns the sorted matrix back
def get_quasi_diag(link):
    link = link.astype(int)
    # get the first and the second item of the last tuple
    sort_ix = pd.Series([link[-1,0], link[-1,1]])
    # the total num of items is the third item of the last list
    num_items = link[-1, 3]
    # if the max of sort_ix is bigger than or equal to the max_items
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
        df0 = sort_ix[sort_ix >= num_items] 
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j,0] # item 1
        df0 = pd.Series(link[j, 1], index=i+1)
        sort_ix = pd.concat([sort_ix,df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

# Call the function to get the sorted matrix
sort_ix = get_quasi_diag(link)
stocks_compl = np.array(stocks_compl)
df_vis = returns[stocks_compl[sort_ix]]
corr2 = df_vis.corr()
# Plot the sorted matrix
ax = sns.heatmap(corr2, cmap="coolwarm")

# Make a function to get the cluster variance
def get_cluster_var(cov, c_items):
    cov_ = cov.iloc[c_items, c_items] # matrix slice
    # calculate the inversev-variance portfolio
    ivp = 1./np.diag(cov_)
    ivp/=ivp.sum()
    w_ = ivp.reshape(-1,1)
    c_var = np.dot(np.dot(w_.T, cov_), w_)[0,0]
    return c_var

# ********************************************************************* RECURSUVE BISECTION ***************************************************************************

# Make a function to generate the weight matrix using recursive bisection
def get_rec_bipart(cov, sort_ix):
    # compute HRP allocation
    # intialize weights of 1
    w = pd.Series(1, index=sort_ix)
    # intialize all items in one cluster
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[int(j):int(k)] for i in c_items for j,k in
        ((0,len(i)/2),(len(i)/2,len(i))) if len(i)>1]
        # now it has 2
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i] # cluster 1
            c_items1 = c_items[i+1] # cluter 2
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0/(c_var0+c_var1)
            w[c_items0] *= alpha
            w[c_items1] *=1-alpha
    return w

#************************************************************* METHODS TO COMPARE THE STRATEGY ************************************************************************

# Compute the weights according to 'minimum variance' method 
def compute_MV_weights(covariances):
    inv_covar = np.linalg.inv(covariances)
    u = np.ones(len(covariances))
    x = np.dot(inv_covar, u) / np.dot(u, np.dot(inv_covar, u))
    return pd.Series(x, index = stocks_compl, name="MV")

# Compute the weights according to 'risk parity' method
def compute_RP_weights(covariances):
    weights = (1 / np.diag(covariances))
    x = weights / sum(weights)
    return pd.Series(x, index = stocks_compl, name="RP")

# Compute the weights according to 'uniform weights' method
def compute_unif_weights(covariances):
    x = [1 / len(covariances) for i in range(len(covariances))]
    return pd.Series(x, index = stocks_compl, name="unif")

cov = returns.cov()

# Store the weights generated by different methods
weights_HRP = get_rec_bipart(cov, sort_ix)
new_index = [returns.columns[i] for i in weights_HRP.index]
weights_HRP.index = new_index
weights_HRP.name = "HRP"

weights_MV = compute_MV_weights(cov)
weights_RP = compute_RP_weights(cov)
weights_unif = compute_unif_weights(cov)

#*******************************************************VARIOUS PARAMETERS TO COMPARE DIFFERENT METHODS****************************************************************

# Make a results dataframe containing results from all the methods 
results = weights_HRP.to_frame()
results = results.join(weights_MV.to_frame())
results = results.join(weights_RP.to_frame())
results = results.join(weights_unif.to_frame())

# Print the results dataframe
print(results)

# Make a function to compute the expected returns which takes the weight matrix as an argument
def compute_ER(weights):
    mean = returns.mean(0)
    return weights.values * mean

# Pass the weight matrix generated by different methods to the 'compute_ER' function 
# to get the expected returns 
er_hrp = compute_ER(weights_HRP)
er_hrp.name = "HRP"
er_mv = compute_ER(weights_MV)
er_mv.name = "MV"
er_rp = compute_ER(weights_RP)
er_rp.name = "RP"
er_unif = compute_ER(weights_unif)
er_unif.name = "unif"

# Make a seperate dataframe to store the expected returns generated by all the methods used 
ers = er_hrp.to_frame()
ers = pd.concat([ers,er_mv.to_frame()])
ers = pd.concat([ers,er_rp.to_frame()])
ers = pd.concat([ers,er_unif.to_frame()])
ers = ers.sum()
ers.name = "Expected Return"
ers = ers.to_frame()

# Print the expected returns dataframe
print(ers)

# Make a function to calculate the volitality of the portfolio generated
# by different methods which takes the weight matrix and covariance matrix as arguments 
def portfolio_volatility(weights, cov):
    return np.sqrt(np.dot(np.dot(weights.values, cov.values), weights.values))

# Find the volitality of the various portfolio constructed by different methods
data = [portfolio_volatility(weights_HRP, cov)]
data.append(portfolio_volatility(weights_MV, cov))
data.append(portfolio_volatility(weights_RP, cov))
data.append(portfolio_volatility(weights_unif, cov))
volatility = pd.DataFrame(data = data, index=["HRP", "MV", "RP", "unif"],
columns=["Volatility"])

# Assume the risk free return rate to be zero
def risk_free():
    return 0

# Make a function to calculate the sharpe ratio of all the portfolios 
# which takes the weight matrix and covariance matrix as arguments
def sharpe_ratio(weights, cov):
    ret_portfolio = compute_ER(weights).sum()
    ret_free = risk_free()
    volatility = portfolio_volatility(weights, cov)
    return (ret_portfolio - ret_free)/volatility

# Pass the weight and covariance matrix to the 'sharpe_ratio' function 
data = [sharpe_ratio(weights_HRP, cov)]
data.append(sharpe_ratio(weights_MV, cov))
data.append(sharpe_ratio(weights_RP, cov))
data.append(sharpe_ratio(weights_unif, cov))
sharpe_R= pd.DataFrame(data = data, index=["HRP", "MV", "RP", "unif"],
columns=["Sharpe Ratio"])

# Print the Sharpe ratio dataframe 
print(sharpe_R)

# Make a function to compute the maximum drawdown
def compute_mdd(weights):
    df = weights * prices
    df = df.sum(1)
    roll_max = df.cummax()
    daily_drawdown = df/roll_max - 1.0

# Pass the weights matrix to 'compute_mdd' function and make a 
# seperate dataframe 'dd' to store them
data = [compute_mdd(weights_HRP)]
data.append(compute_mdd(weights_MV))
data.append(compute_mdd(weights_RP))
data.append(compute_mdd(weights_unif))
dd = pd.DataFrame(data = data, index=["HRP", "MV", "RP", "unif"],
columns = ["Max DD"])

# Make a function to compute the diversification ratio for all the portfolios
def diversification_ratio(weights, cov):
    p_volatility = portfolio_volatility(weights, cov)
    return np.dot(np.sqrt(np.diag(cov.values)), weights) / p_volatility

# Pass the weights and covariance matrix to the 'diversification_ratio' function 
# and store the results in a seperate dataframe 'dr'
data = [diversification_ratio(weights_HRP, cov)]
data.append(diversification_ratio(weights_MV, cov))
data.append(diversification_ratio(weights_RP, cov))
data.append(diversification_ratio(weights_unif, cov))
dr = pd.DataFrame(data = data, index=["HRP", "MV", "RP", "unif"],
columns = ["Div Ratio"])

#************************************************************************* FINAL RESULTS *****************************************************************************

# Join all the dataframes constructed in one dataframe to get final results 
final_results = ers.join(volatility)
final_results = final_results.join(sharpe_R)
final_results = final_results.join(dd)
final_results = final_results.join(dr)
print(final_results)
