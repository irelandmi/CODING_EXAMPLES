#%%

##############################
#run the "live data.py" file before running this script.
#this file will display a live matplotlib graph of the data
##############################



import numpy as np
import json
from datetime import datetime
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import csv
import pandas as pd
#%%
data = pd.read_csv('prices_flat.csv')

#%%
plt.style.use('fivethirtyeight')

def animate(i):
    data = pd.read_csv('prices_flat.csv')
    x = data.index[-1000:]
    y1 = data['price'].iloc[-1000:]

    plt.cla()
    plt.plot(x, y1, label='Price')
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval=100)

plt.tight_layout()
plt.show()
