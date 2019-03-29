import numpy as np 
import matplotlib.pyplot as plt 


x=np.arange(-10.0,10.0,1)

# liinear equation 
y=2*(x)+10
# non linear equation
y_non_linear=1*(x**3)+1*(x**2)+1*x+3


y_intercept=np.random.normal(size=x.size)

#linear but intercept different and non-linear
y_new=y+y_intercept


#non linear and intercept different
y_nonlinear=y_non_linear+y_intercept 



# linear
plt.plot(x,y,'-bo')
plt.plot(x,y_new,'-or')
plt.xlabel("x")
plt.ylabel('y')


# nonlinear
plt.plot(x,y_non_linear,'-go')
plt.plot(x,y_nonlinear,'-oy')
plt.xlabel("x")
plt.ylabel('y')

plt.show()






