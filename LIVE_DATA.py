#%%

import numpy as np
from binance.client import Client
import json
from datetime import datetime
from binance.websockets import BinanceSocketManager
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import csv


if __name__ == "__main__":
    client = Client()
    live_data = []
    counter = 1

    field_names = ["price"]

    with open("prices_flat.csv", 'w') as flat_file:
        csv_writer = csv.DictWriter(flat_file, fieldnames=field_names)
        csv_writer.writeheader()


        

    def process_message(msg):
        #print("message type: {}".format(msg['e']))
        
        global counter
        event_t = datetime.fromtimestamp(msg['E'] / 1e3)
        trade_t = datetime.fromtimestamp(msg['T'] / 1e3)
        live_data.append(msg["p"])
        counter += 1
        with open("prices_flat.csv", "a") as flat_file:
            csv_writer = csv.DictWriter(flat_file,fieldnames=field_names)
            info = {"price":msg["p"]}
            csv_writer.writerow(info)
        

        
    bm = BinanceSocketManager(Client)
    # start any sockets here, i.e a trade socket
    #conn_key = bm.start_trade_socket('BTCUSDT', process_message)
    conn_key = bm.start_aggtrade_socket('BTCUSDT', process_message)
    # then start the socket manager
    bm.start()
    size = 100

    line1 = []
    time.sleep(1000)
    bm.close()
    print("complete")
