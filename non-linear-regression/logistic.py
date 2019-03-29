import numpy as np 
import matplotlib.pyplot as plt 

x=np.arange(-5,5,.1)

y=1-4/(1+np.power(3, x-2))

plt.plot(x,y,'-oy')
plt.show()


