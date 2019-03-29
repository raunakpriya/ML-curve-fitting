import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(12344)

x_train = np.linspace(0, 1, 10)
noise_10 = np.random.normal(0, 0.3, 10)
y_train = np.sin(2*np.pi*x_train) + noise_10

x_test = np.linspace(0, 1, 100)
noise_100 = np.random.normal(0, 0.3, 100)
y_test = np.sin(2*np.pi*x_test) + noise_100

x_1000 = np.linspace(0, 1, 1000)

z0=np.polyfit(x_train, y_train, 0)
p0=np.poly1d(z0)
p0(x_test)

z3=np.polyfit(x_train, y_train, 3)
p3=np.poly1d(z3)
p3(x_test)
z5=np.polyfit(x_train, y_train, 5)
p5=np.poly1d(z5)
p5(x_test)
z5=np.polyfit(x_train, y_train, 5)
p5=np.poly1d(z5)
p5(x_test)
plt.scatter(x_train, y_train)
plt.scatter(x_train, y_train)
plt.plot(x_train,p0(x_train))
plt.plot(x_train,p3(x_train))
plt.plot(x_train,p5(x_train))
plt.plot(x_train,p9(x_train))

plt.scatter(x_train,y_train)
plt.plot(x_test,p0(x_test))
plt.plot(x_test,p3(x_test))
plt.plot(x_test,p5(x_test))
plt.plot(x_test,p9(x_test))

e0=mean_squared_error(y_train,p0(x_train))
e3=mean_squared_error(y_train,p3(x_train))
e5=mean_squared_error(y_train,p5(x_train))
e9=mean_squared_error(y_train,p9(x_train))

error=[e0,e3,e5,e9]
error
plt.plot(error)


e0=mean_squared_error(y_test,p0(x_test))
e3=mean_squared_error(y_test,p3(x_test))
e5=mean_squared_error(y_test,p5(x_test))
e9=mean_squared_error(y_test,p9(x_test))

error1=[e0,e3,e5,e9]
plt.plot(error1)

x_train = x_train[:, np.newaxis]
x_test = x_test[:, np.newaxis]

low_alpha = make_pipeline(PolynomialFeatures(degree=9),Ridge(alpha=np.exp(-18)))
low_alpha.fit(x_train, y_train)

plt.figure(1)
plt.plot(x_test, low_alpha.predict(x_test), label= 'ln alpha = -18')
plt.plot(x_1000, np.sin(2*np.pi*x_1000))
plt.legend()
plt.show

