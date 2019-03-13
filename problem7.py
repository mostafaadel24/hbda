# problem 7
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import metrics

x = np.array([0, 10, 20, 30, 40,  50, 60, 70, 80, 90])
print(x) #no. of ads
p = np.array([0.002, 0.007, 0.027, 0.092, 0.27, 0.576, 0.833, 0.948, 0.985, 0.996])# %who bought 
plt.scatter(x,p)
plt.title('percentage of how bought vs no. of ads')
plt.xlabel('no. of ads')
plt.ylabel('percentage of how bought')

#linear regression
linearReg = linear_model.LinearRegression()
linearReg.fit(x.reshape(-1,1),p.reshape(-1,1))
pred_linear = linearReg.predict(x.reshape(-1,1))
MSE_linear = metrics.mean_squared_error(p, pred_linear)
R2_linear = metrics.r2_score(p, pred_linear)
print('linear: MSE =',MSE_linear,'R2 =', R2_linear)
plt.plot(x,pred_linear,color='r')

x1 = np.array([ 10, 20, 30, 40,  50, 60, 70, 80, 90])
p = np.array([ 0.007, 0.027, 0.092, 0.27, 0.576, 0.833, 0.948, 0.985, 0.996])# %who bought 
#power regression
powerReg = linearReg.fit(np.log(x1).reshape(-1,1),np.log(p).reshape(-1,1))
pred_power = np.exp(powerReg.predict(np.log(x1).reshape(-1,1)))
MSE_power = metrics.mean_squared_error( p, pred_power)
R2_power = metrics.r2_score( p, pred_power)
print('power: MSE =',MSE_power,'R2 =', R2_power)
plt.plot(x1,pred_power,color='g')

#exponential regression
exponentialReg = linearReg.fit(x1.reshape(-1,1),np.log(p).reshape(-1,1))
pred_exponential = np.exp(exponentialReg.predict(x1.reshape(-1,1)))
MSE_exponential = metrics.mean_squared_error(p, pred_exponential)
R2_exponential = metrics.r2_score(p, pred_exponential)
print('exponential: MSE =',MSE_exponential,'R2 =', R2_exponential)
plt.plot(x1,pred_exponential,color='y')

#logarithmic regression
logarithmicReg = linearReg.fit(np.log(x1).reshape(-1,1),p.reshape(-1,1))
pred_logarithmic = logarithmicReg.predict(np.log(x1.reshape(-1,1)))
MSE_logarithmic = metrics.mean_squared_error(p, pred_logarithmic)
R2_logarithmic = metrics.r2_score(p, pred_logarithmic)
print('logarithmic: MSE =',MSE_logarithmic,'R2 =', R2_logarithmic)
plt.plot(x1,pred_logarithmic,color='k')

#logistic regression
logisticReg = linearReg.fit(x1.reshape(-1,1),
                            np.log(1/p-1).reshape(-1,1))
z = np.exp(logisticReg.predict(x1.reshape(-1,1)))
pred_logistic = 1/(1+z) 
MSE_logistic = metrics.mean_squared_error(np.log(p/(1-p)), np.log(pred_logistic/(1-pred_logistic)))
R2_logistic = metrics.r2_score(p, pred_logistic)
print('logistic: MSE =',MSE_logistic,'R2 =', R2_logistic)
predictionOf100 = np.exp(logisticReg.predict(np.array([100]).reshape(-1,1)))
predictionOf100 =1/(1+predictionOf100)
print('predrction of 100 =',predictionOf100)
plt.plot(x1,pred_logistic,color='m')
plt.show()
