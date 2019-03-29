import numpy as np 
import matplotlib.pyplot as plt 

x=np.arange(-5,5,.1)

y=np.power(x,2)

plt.plot(x,y,'-oy')
plt.show()