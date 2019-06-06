from pandas import *

raw_X = pandas.read_csv('dengue-ml-features.data.csv', sep=',')
fixed = raw_X.fillna(raw_X.mean())
fixed.to_csv("dengue-ml-features-fixed.data.csv", header=True)