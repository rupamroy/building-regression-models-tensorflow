import numpy as np

from returns_data import read_goog_sp500_dataframe

returns = read_goog_sp500_dataframe()

returns["Intercept"] = 1

print(np.array(returns[["SP500", 'Intercept']][1:-1]))

