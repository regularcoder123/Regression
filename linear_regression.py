import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score

#data preprocessing
dia = datasets.load_diabetes()
print(dia.feature_names)
dia_x,dia_y = datasets.load_diabetes(return_X_y=True)
dia_x = dia_x[:,np.newaxis,2]
dia_x_train = dia_x[:80]
dia_x_test = dia_x[80:]

dia_y_train = dia_y[:80]
dia_y_test = dia_y[80:]

#model
reg = linear_model.LinearRegression()
reg.fit(dia_x_train,dia_y_train)
dia_y_pred = reg.predict(dia_x_test)
print("Coeficcients: ",reg.coef_)
print("Mean Squared Error:%.2f"%mean_squared_error(dia_y_test,dia_y_pred))
print("Coeficcient of Determination:%.2f"%r2_score(dia_y_test,dia_y_pred))

#plot
plt.scatter(dia_x_test,dia_y_test,color='black')
plt.plot(dia_x_test,dia_y_pred,color='blue',linewidth = 3)

plt.xticks()
plt.yticks()
plt.show()