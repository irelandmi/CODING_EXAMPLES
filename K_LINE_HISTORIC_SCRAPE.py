#%%
import pandas as pd
import numpy as np
from binance.client import Client
import json
import datetime
import psycopg2
import time

client = Client()
#connection parameters
connection = psycopg2.connect(user="postgres",
                                  password="password",
                                  host="localhost",
                                  port="5432",
                                  database="datawarehouse_local")

# Create a cursor to perform database operations
cursor = connection.cursor()
# Print PostgreSQL details
print("PostgreSQL server information")
print(connection.get_dsn_parameters(), "\n")
cursor.execute("SELECT version();")
record = cursor.fetchone()
print("You are connected to - ", record, "\n")
cursor.close()

#set symbol to return K Lines for
symbol_q = "BNBUSDT"

#variables for start/stop points of k lines and the chunks of time we wish to query
start = datetime.datetime(year=2018,month=11,
day=1,hour=0,minute=0,second=0)
tdelta = datetime.timedelta(hours=8)
end = start + tdelta
stop_point = datetime.datetime(year=2021,month=1,
day=1,hour=0,minute=0,second=0)

while start < stop_point:
    #reduce query speed as to not spam the API
    time.sleep(0.2)
    #open a new cursor
    cursor = connection.cursor()
    #query api for symbol at a set interval of time
    testline = client.get_historical_klines(symbol=symbol_q,interval="1m",start_str=str(start),end_str=str(end))
    testline = pd.DataFrame(testline)
    try:
        #Convert start and end timestamps to datetimePostgreSQL datetime variable
        testline.iloc[:,0] = pd.to_datetime(testline.iloc[:,0]*1000000)
        testline.iloc[:,6] = pd.to_datetime(testline.iloc[:,6]*1000000)
    except:
        #occasional empty dataframe due to server outages on the API side
        pass

    for ind, row in testline.iterrows():
        try:
            #extract and declare variables for inserting into postgres
            open_stamp = row[0]
            symbol=symbol_q
            open_v=float(row[1])
            high=float(row[2])
            low=float(row[3])
            close=float(row[4])
            num_trades=int(row[8])
            vol=float(row[5])
            close_stamp = row[6]

            #print(open_stamp)
            #print(symbol)
            #print(open_v)
            #print(high)
            #print(low)
            #print(num_trades)

            #insert query with string formating variables for insert
            postgres_insert_query = """ INSERT INTO public."K_LINES" 
            (open_stamp, symbol, open, high, low, close, num_trades, volume, close_stamp) 
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (open_stamp, symbol) DO NOTHING;"""
            
            #format query and insert into postgres
            record_to_insert = (open_stamp, symbol, open_v, high ,low ,close, num_trades, vol, close_stamp)
            cursor.execute(postgres_insert_query, record_to_insert)
        except:
            pass
        
    #commit queries and close connection
    connection.commit()
    cursor.close()
    #print status
    print(start)
    print(end)
    start = start + tdelta
    end = end + tdelta
