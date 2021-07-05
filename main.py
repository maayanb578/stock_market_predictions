import matplotlib.pyplot as plt

from utils import YahooFinanceHistory
from all_models import *
from scraping_yahoo_data import scrape_quotes

def clean_df(df):
    # dropping ALL duplicate values
    df.drop_duplicates(subset=['Date'], keep=False, inplace=True)
    # drop unnecessary columns
    columns_valid = (set(list(df.columns)) - set(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']))
    if columns_valid != set():
        for col in columns_valid:
            del df[col]
    # if mor than 10% of the values is null scrape again
    null_rows = df.isnull().sum().sum()
    if null_rows > 1000:
        symbol = 'AAPL'
        scrape_quotes(symbol)

# displaying data
# print(data)


if __name__ == "__main__":
    # Get data
    # df = YahooFinanceHistory('AAPL', days_back=15000).get_quote()
    # df =  pd.read_csv('value.txt')
    symbol = 'AAPL'
    scrape_quotes(symbol)
    model_mae_scores = {}
    df = pd.read_csv('AAPL.csv')
    clean_df(df)
    dt_mean_ava_err = decision_tree(df)
    model_mae_scores['decision_tree'] = dt_mean_ava_err
    df = pd.read_csv('stocks_data_30_6.csv')
    naive_mean_ava_err = naive(df)
    model_mae_scores['naive'] = naive_mean_ava_err['naive']
    sma20_mean_err = ma20(df)
    model_mae_scores['sma20'] = sma20_mean_err
    dense_mean_err = dense(df)
    model_mae_scores['dense'] = dense_mean_err

    mae_series = pd.Series(model_mae_scores)
    order = mae_series.sort_values()
    sns.barplot(x=order.values, y=order.index, orient='h')

    plt.xlabel('Mean Absolute Error')
    plt.xticks(rotation='vertical', fontsize=14)
    plt.title('Mean Average Error of All Models Tested')
    plt.show()
    print("Done")



