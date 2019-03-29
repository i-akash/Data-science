import numpy as np 
import matplotlib.pyplot as plt 

x=np.arange(-5,5,.1)

y=np.log(x)

plt.plot(x,y,'-oy')
plt.show()