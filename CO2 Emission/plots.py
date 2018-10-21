import pandas as pd
import matplotlib.pyplot as plt

# Link to download the CSV file is "https://app.box.com/s/ln8oltpfktrnlg0jxpnac5iz7v4ln7so"
df = pd.read_csv("E:/Pythoncode/Coursera/CO2 Emission/FuelConsumption.csv")
#print(df.describe())

# choosing data only on these following four rows into new dataframe "cdf"
# hist -> histogram
cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
cdf.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2Emission")
plt.show()
