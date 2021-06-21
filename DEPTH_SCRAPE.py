#%%

from binance.client import Client
import json
from datetime import datetime
import psycopg2

client = Client()
connection = psycopg2.connect(user="pi",
                                  password="password",
                                  host="raspberrypi",
                                  port="5432",
                                  database="MONEY_TREE")

# Create a cursor to perform database operations
cursor = connection.cursor()
# Print PostgreSQL details
print("PostgreSQL server information")
print(connection.get_dsn_parameters(), "\n")
cursor.execute("SELECT version();")
record = cursor.fetchone()
print("You are connected to - ", record, "\n")

#format insert query
postgres_insert_query = """ INSERT INTO public."DEPTH" (update_id,value,side,volume,"timestamp",symbol) VALUES (%s,%s,%s,%s,%s,%s)"""
symbol_list = ["BTCUSDT","LTCUSDT","ETHUSDT","BCHUSDT"]

#query current order book for given symbol and insert into postgres
for ii in symbol_list:
    symbol_q = ii
    depth = client.get_order_book(symbol=symbol_q, limit=5)
    now = datetime.now()
    update_id = depth["lastUpdateId"]
    bids = depth["bids"]
    asks = depth["asks"]
    type_pair = "bids"
    for i in bids:
        #value
        a=float(i[0])
        #volume
        b=float(i[1])
        print(a)
        print(b)
        record_to_insert = (update_id, a, type_pair, b ,now ,symbol_q)
        cursor.execute(postgres_insert_query, record_to_insert)
    type_pair = "asks"
    for i in asks:
        #value
        a=float(i[0])
        #volume
        b=float(i[1])
        print(a)
        print(b)
        record_to_insert = (update_id, a, type_pair, b ,now ,symbol_q)
        cursor.execute(postgres_insert_query, record_to_insert)

#commit queries and close connections
connection.commit()
cursor.close()
connection.close()

