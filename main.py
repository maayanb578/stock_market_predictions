import matplotlib.pyplot as plt

from utils import YahooFinanceHistory
from all_models import *

# data = pd.read_csv("stock_data5.csv")


# sorting by first name
# data.sort_values("Date", inplace=True)

# dropping ALL duplicate values
# data.drop_duplicates(subset=['Date'], keep=False, inplace=True)

# displaying data
# print(data)


if __name__ == "__main__":
    # Get data
    # df = YahooFinanceHistory('AAPL', days_back=15000).get_quote()
    model_mae_scores = {}
    df = pd.read_csv('stocks_data_30_6.csv',parse_dates=True, index_col=0)
    dt_mean_ava_err = decision_tree(df)
    model_mae_scores['decision_tree'] = dt_mean_ava_err
    df = pd.read_csv('stocks_data_30_6.csv')
    naive_mean_ava_err = naive(df)
    model_mae_scores['naive'] = naive_mean_ava_err['naive']

    mae_series = pd.Series(model_mae_scores)
    order = mae_series.sort_values()
    sns.barplot(x=order.values, y=order.index, orient='h')
    plt.xlabel('Mean Absolute Error')
    plt.xticks(rotation='vertical', fontsize=14)
    plt.title('Mean Average Error of All Models Tested')
    plt.show()
    print("Done")


