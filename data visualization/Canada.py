import pandas as pd 

df=pd.read_excel('./Canada.xlsx',skiprows=range(20),skipfooter=2)

print(df.index())