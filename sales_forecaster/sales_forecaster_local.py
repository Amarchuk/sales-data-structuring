from __future__ import print_function
import os.path
import glob
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

import parser
import gservice


def get_liquidation_orders(orders_df, liquidataion_limit_df):
    try:
        orders_with_liquidation_limit = pd.merge(orders_df, liquidataion_limit_df,
                                                 how='left',
                                                 on=['Cin7', 'Year', 'Month'])

        nan_limit = orders_with_liquidation_limit[orders_with_liquidation_limit['Price Limit'].isnull()]
        if nan_limit.shape[0] > 0:
            print('Liquidation limit is not available for these orders:\n', nan_limit)

        orders_with_liquidation_limit.dropna(subset=['Price Limit'], inplace=True)

        orders_liquidation = orders_with_liquidation_limit[
            orders_with_liquidation_limit['Price/Qty'] <= orders_with_liquidation_limit['Price Limit']
            ]

        orders_liquidation = orders_liquidation.drop(['Liquidation Limit', 'Normal Price', 'Price Limit'], axis=1)
        return orders_liquidation
    except KeyError:
        print("Could not match orders with liquidation limits. It may not have found liquidation limits.")
        return pd.DataFrame()


def add_out_of_stock_days(orders_df, out_of_stock_df):
    try:
        orders_with_out_of_stock_days = pd.merge(
            orders_df,
            out_of_stock_df,
            how='left',
            on=['Cin7', 'Year', 'Month', 'Market Place'])
        orders_with_out_of_stock_days.fillna(0, inplace=True)

        return orders_with_out_of_stock_days
    except KeyError:
        print("Could not add stock out days.")
        return orders_df


def format_calculations_for_output(df, cin7_product, out_of_stock, sales_channel, sales_type):
    output = add_out_of_stock_days(df, out_of_stock)

    output = match_cin7_product(output, cin7_product)

    output['Sales Type'] = sales_type
    output['Sales Channel'] = sales_channel
    output = output.rename(columns={'Market Place': 'Country'})

    if (sales_type == 'PPC') | (sales_type == 'Organic'):
        output['Revenue'] = output['Qty'] * output['Avg Sale Price']
        output['Date'] = pd.to_datetime(
            output['Year'].astype(str) + ' ' + output['Month'].astype(str),
            format='%Y %B').dt.strftime('%m/%d/%Y')
        output = output[['Brand', 'Country', 'Sales Channel', 'Product Group', 'Cin7',
                         'Sales Type', 'Date', 'Year', 'Month', 'Qty',
                         'Out of stock days', 'Avg Sale Price', 'Revenue']]
    else:
        output['Revenue'] = output['Qty'] * output['Price/Qty']
        output['Date'] = pd.to_datetime(
            output['Year'].astype(str).apply(lambda l: l[:-2]) + ' ' + output['Month'] + ' ' + output['Day'].astype(str).apply(lambda l: l[:-2]),
            format='%Y %B %d').dt.strftime('%m/%d/%Y')
        output = output[['Brand', 'Country', 'Sales Channel', 'Product Group', 'Cin7',
                         'Sales Type', 'Date', 'Year', 'Month', 'Day', 'Qty',
                         'Out of stock days', 'Price/Qty', 'Revenue']]

    return output


def match_asin_cin7(df, asin_cin7_map, duplication_method):
    try:
        matched = pd.merge(df, asin_cin7_map,
                           how='left',
                           left_on='ASIN',
                           right_on='Amazon-ASIN')

        nan_matched = matched[matched['Cin7'].isnull()]
        if nan_matched.shape[0] > 0:
            print('Did not found cin7 for the following ASINs:\n', nan_matched['ASIN'].unique())
        matched.dropna(subset=['Cin7'], inplace=True)

        matched.drop(['Amazon-ASIN', 'ASIN'], axis=1, inplace=True)

        if duplication_method == 'out-of-stock':
            matched = matched.drop_duplicates(subset=['Market Place', 'Cin7', 'Year', 'Month'])
        elif duplication_method == 'sales':
            matched = matched.drop_duplicates()
        elif duplication_method == 'orders':
            pass

        return matched
    except KeyError:
        print('Could not match the ASIN to the orders.')
        return df


def match_cin7_product(df, cin7_product_map):
    try:
        matched = pd.merge(df, cin7_product_map,
                           how='left',
                           on='Cin7')
        columns_to_unique = list(matched.columns)
        columns_to_unique.remove('Brand')
        columns_to_unique.remove('Product Group')
        matched = matched.drop_duplicates(subset=columns_to_unique)

        nan_matched = matched[matched['Brand'].isnull() | matched['Product Group'].isnull()]
        if nan_matched.shape[0] > 0:
            print('Did not found Brand or Product Group for the following Cin7s:\n', nan_matched['Cin7'])

        return matched
    except KeyError:
        print('Could not match the product details to the orders.')
        return df


def calculate_historical_table(df):
    qty_sum = df.groupby([
        'Year', 'Month', 'Day', 'Market Place', 'Cin7'
    ])['Qty'].sum()
    unit_price_mean = df.groupby([
        'Year', 'Month', 'Day', 'Market Place', 'Cin7'
    ])['Price/Qty'].mean()

    calc_historical = pd.concat([qty_sum, unit_price_mean], axis=1).reset_index()

    calc_historical = calc_historical[['Cin7', 'Market Place', 'Year', 'Month', 'Day', 'Qty', 'Price/Qty']]
    return calc_historical


def sum_ppc_orders_by_product_group(df):
    qty_sum = df.groupby([
        'Market Place', 'Year', 'Month', 'Brand', 'Product Group'
    ])['PPC Orders'].sum()
    ppc_sums = qty_sum.reset_index()
    return ppc_sums


def calculate_ppc_portions(df):
    monthly_brand_pg_sum = df.groupby([
        'Market Place', 'Year', 'Month', 'Brand', 'Product Group'
    ])['Qty'].sum().reset_index().rename(columns={'Qty': 'Category Sum'})

    try:
        monthly_cin7_sum = df.groupby([
            'Cin7', 'Market Place', 'Year', 'Month', 'Brand', 'Product Group'
        ])['Qty'].sum().reset_index().rename(columns={'Qty': 'Product Sum'})

        df_with_brand_pg_sum = pd.merge(monthly_cin7_sum, monthly_brand_pg_sum,
                                        how='left',
                                        on=['Market Place', 'Year', 'Month', 'Brand', 'Product Group'])

        df_with_brand_pg_sum['Portion'] = df_with_brand_pg_sum['Product Sum'] / df_with_brand_pg_sum['Category Sum']
        df_with_brand_pg_sum.fillna(0, inplace=True)

        return df_with_brand_pg_sum[['Cin7', 'Market Place', 'Year', 'Month', 'Portion']]
    except KeyError:
        print('Could not calculate the distribution of PPC Orders.')
        return df


def reallocate_ppc_qty(ppc_organic, sales_ppc, portion):
    try:
        monthly_ppc_organic_sum = (ppc_organic.groupby(
            ['Cin7', 'Market Place', 'Year', 'Month', 'Brand', 'Product Group'], as_index=False)
              .agg({'Qty': 'sum', 'Price/Qty': 'mean'})
              .rename(columns={'Qty': 'Product Sum', 'Price/Qty': 'Avg Sale Price'}))

        monthly_ppc_organic_sum = pd.merge(monthly_ppc_organic_sum, portion,
                               how='left',
                               on=['Cin7', 'Market Place', 'Year', 'Month'])
        monthly_ppc_organic_sum = pd.merge(monthly_ppc_organic_sum, sales_ppc,
                               how='left',
                               on=['Market Place', 'Year', 'Month', 'Brand', 'Product Group'])

        monthly_ppc_organic_sum['Date'] = monthly_ppc_organic_sum\
            .apply(lambda row: datetime.strptime(str(row['Year']) + row['Month'], '%Y%B'), axis=1)
        monthly_ppc_organic_sum = monthly_ppc_organic_sum.sort_values(
            ['Cin7', 'Date'], ascending=[True, True]).reset_index(drop=True)
        monthly_ppc_organic_sum['Portion'] = monthly_ppc_organic_sum.groupby(
            ['Cin7', 'Market Place'])['Portion']\
            .rolling(2, min_periods=1).mean()\
            .reset_index(drop=True)

        monthly_ppc_organic_sum['PPC Orders'] = monthly_ppc_organic_sum['PPC Orders'] * \
                                                            monthly_ppc_organic_sum['Portion']

        monthly_ppc_organic_sum['Organic Orders'] = monthly_ppc_organic_sum['Product Sum'] - \
                                                    monthly_ppc_organic_sum['PPC Orders']
        monthly_ppc_organic_sum = monthly_ppc_organic_sum.round()

        return monthly_ppc_organic_sum[['Cin7', 'Market Place', 'Year', 'Month',
                                        'Product Sum', 'Avg Sale Price', 'PPC Orders', 'Organic Orders']]
    except KeyError:
        print('Could not reallocate the PPC Orders.')
        return ppc_organic


def summarize_by_sales_type(df, cin7_product_map, sales_type):
    try:
        summarized = match_cin7_product(df, cin7_product_map)
        qty_sum = summarized.groupby([
            'Brand', 'Market Place', 'Product Group', 'Cin7', 'Year', 'Month'
        ])['Qty'].sum()
        price_avg = summarized.groupby([
            'Brand', 'Market Place', 'Product Group', 'Cin7', 'Year', 'Month'
        ])['Price/Qty'].mean()

        summarized = pd.concat([qty_sum, price_avg], axis=1).reset_index() \
            .rename(columns={'Qty': 'Sales QTY', 'Price/Qty': 'Avg Sale Price'})
        summarized['Revenue'] = summarized['Sales QTY'] * summarized['Avg Sale Price']
        summarized['Date'] = pd.to_datetime(summarized['Year'].astype(str).apply(lambda l: l[:-2]) + ' ' + summarized['Month'],
                                            format='%Y %B').dt.strftime('%m/%d/%Y')
        summarized['Sales Type'] = sales_type
        # TODO: is this line correct?
        summarized['Sales Channel'] = 'Amazon' if sales_type != 'Shopify' and sales_type != 'Wholesale' else 'Non-Amazon'
        return summarized
    except KeyError:
        print('Cannot summarize dataframe: ', df)
        return df


def summarize_reallocated_sales_type(df, cin7_product_map, sales_type):
    try:
        summarized = match_cin7_product(df, cin7_product_map).rename(columns={'Qty': 'Sales QTY'})

        summarized['Revenue'] = summarized['Sales QTY'] * summarized['Avg Sale Price']
        summarized['Date'] = pd.to_datetime(summarized['Year'].astype(str) + ' ' + summarized['Month'],
                                            format='%Y %B').dt.strftime('%m/%d/%Y')
        summarized['Sales Type'] = sales_type
        summarized['Sales Channel'] = 'Amazon'
        return summarized
    except KeyError:
        print('Cannot summarize dataframe: ', df)
        return df


def main(orders_regex, out_of_stock_regex, sales_regex, input_regex):

    order_files = glob.glob(orders_regex)
    stock_out_files = glob.glob(out_of_stock_regex)
    sales_files = glob.glob(sales_regex)

    input = glob.glob(input_regex)

    cin7_product = pd.read_excel(input[0], sheetname='Input-Cin7-Product-Map')
    asin_cin7 = pd.read_excel(input[0], sheetname='Input-ASIN-Cin7-Map')

    liq_limits = pd.read_excel(input[0], sheetname='Input-Liquidation-Limits')
    # liq_limits = pd.read_excel(sales_files[0], sheetname='Input-FT-Std. Price')
    print('Read succsesfully')
    liquidation_limit = parser.parse_liquidation_limits(liq_limits)
    # liquidation_limit = parser.parse_liquidation_limits_std(liq_limits)

    out_of_stock = parser.read_out_of_stock_csv(stock_out_files)
    out_of_stock = match_asin_cin7(out_of_stock, asin_cin7, 'out-of-stock')

    # sales = parser.read_sales_xlsx(sales_files)
    # sales = match_asin_cin7(sales, asin_cin7, 'sales')
    # sales = match_cin7_product(sales, cin7_product)
    # sales_ppc = sum_ppc_orders_by_product_group(sales)

    orders = parser.read_orders_csv(order_files)
    orders = match_asin_cin7(orders, asin_cin7, 'orders')
    orders = orders[['Cin7', 'Year', 'Month', 'Day', 'Market Place', 'Sales Channel',
                     'Qty', 'Price/Qty']]
    orders_amazon = orders[orders['Sales Channel'] != 'Non-Amazon']
    orders_non_amazon = orders[orders['Sales Channel'] == 'Non-Amazon']

    liquidation_orders = get_liquidation_orders(orders_amazon, liquidation_limit)
    calc_historical_liquidation = calculate_historical_table(liquidation_orders)
    sum_liq = summarize_by_sales_type(calc_historical_liquidation, cin7_product, 'Liquidations')

    liquidation_orders_non = get_liquidation_orders(orders_non_amazon, liquidation_limit)
    calc_historical_liquidation_non = calculate_historical_table(liquidation_orders_non)
    sum_liq_non = summarize_by_sales_type(calc_historical_liquidation_non, cin7_product, 'Liquidations')
    sum_liq_non['Sales Channel'] = 'Non-Amazon'

    summarized_output_file = pd.concat([sum_liq, sum_liq_non], ignore_index=True)

    # summarized_output_file = add_out_of_stock_days(summarized_output_file, out_of_stock)
    # summarized_output_file = summarized_output_file.rename(columns={'Market Place': 'Country'})
    # summarized_output_file = summarized_output_file[['Brand', 'Country', 'Sales Channel', 'Product Group', 'Cin7',
    #                                                  'Sales Type', 'Date', 'Year', 'Month', 'Sales QTY',
    #                                                  'Out of stock days', 'Avg Sale Price', 'Revenue']]

    calc_historical_liquidation_formatted = format_calculations_for_output(
        calc_historical_liquidation, cin7_product, out_of_stock, 'Amazon', 'Liquidation'
    )

    calc_historical_liquidation_formatted_non = format_calculations_for_output(
        calc_historical_liquidation_non, cin7_product, out_of_stock, 'Non-Amazon', 'Liquidation'
    )

    liq_summary = pd.concat([calc_historical_liquidation_formatted, calc_historical_liquidation_formatted_non], ignore_index=True)

    liq_summary2 = liq_summary.copy()
    liq_summary2['Revenue'] = liq_summary2['Qty'] * liq_summary2['Price/Qty']
    liq_summary2_1 = liq_summary2.groupby(['Year', 'Month', 'Sales Channel', 'Cin7'])['Qty'].sum()
    liq_summary2_2 = liq_summary2.groupby(['Year', 'Month', 'Sales Channel', 'Cin7'])['Revenue'].sum()
    liq_summary2 = pd.concat([liq_summary2_1, liq_summary2_2], axis=1).reset_index()
    liq_summary2['Avg Sale Price'] = liq_summary2['Revenue']/liq_summary2['Qty']



    print('pass')
    print(liq_summary2.head())


    # with pd.ExcelWriter('calculations.xlsx') as writer:
    #     calc_historical_liquidation_formatted.to_excel(writer, sheet_name='Calc-Historical-Liquidation')

    # with pd.ExcelWriter('liquidations.xlsx') as writer:
    #     liq_summary.to_excel(writer, sheet_name='Historical-Liquidation')

    with pd.ExcelWriter('liquidations_by_day.xlsx') as writer:
        liq_summary.to_excel(writer, sheet_name='Calc-Historical-Liquidation')

    with pd.ExcelWriter('liquidations_by_month.xlsx') as writer:
        liq_summary2.to_excel(writer, sheet_name='Calc-Historical-Liquidation-Month')


#




if __name__ == '__main__':
    main('ORDERS*.csv', 'INVENTORY*.csv', 'SALESPERDAY*.xlsx', '*input.xlsx')
