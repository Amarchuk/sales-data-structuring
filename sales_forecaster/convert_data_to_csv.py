from __future__ import print_function
import os.path
import os
import glob
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

import parser
import gservice
import re

PATH_TO_DATA= '../data/'

def main(orders_regex, out_of_stock_regex, sales_regex, input_regex):

    ABS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../data/'
    print(ABS_PATH)

    order_files = glob.glob(ABS_PATH + orders_regex)
    stock_out_files = glob.glob(ABS_PATH + out_of_stock_regex)
    print(order_files)
    print(stock_out_files)


    # order_files = [f for f in os.listdir(ABS_PATH) if re.match(orders_regex, f)]
    # stock_out_files = [f for f in os.listdir(ABS_PATH) if re.match(out_of_stock_regex, f)]
    # print(order_files)
    # print(stock_out_files)

    for filename in stock_out_files:
        print(filename)
        df = pd.read_excel(filename)
        df.to_csv(filename.replace('.xlsx', '.csv'))
        print('Done')

    for filename in order_files:
        print(filename)
        df = pd.read_excel(filename)
        df.to_csv(filename.replace('.xlsx', '.csv'))
        print('Done')

if __name__ == '__main__':
    main('ORDERS*.xlsx', 'INVENTORY*.xlsx', 'SALESPERDAY*.xlsx', '*input.xlsx')
