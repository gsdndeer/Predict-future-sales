import argparse
import pandas as pd
import numpy as np
import itertools
import time
from xgboost import XGBRegressor


def data_cleaning(sales_train, test):
    # Remove outliers with price > 100000 and sales > 1000
    sales_train = sales_train.drop(sales_train[sales_train['item_price']>100000].index)
    sales_train = sales_train.drop(sales_train[sales_train['item_cnt_day']>1000].index)

    # Replace non-positive price items with the median value 
    median = sales_train[(sales_train['shop_id'] == 32) & (sales_train['item_id'] == 2973) & (sales_train['date_block_num'] == 4) & (sales_train['item_price'] > 0)].item_price.median()
    sales_train.loc[sales_train['item_price'] < 0, 'item_price'] = median

    # Unify duplicated shops
    sales_train.loc[sales_train['shop_id'] == 11,'shop_id'] = 10
    sales_train.loc[sales_train['shop_id'] == 57,'shop_id'] = 0
    sales_train.loc[sales_train['shop_id'] == 58,'shop_id'] = 1
    sales_train.loc[sales_train['shop_id'] == 40,'shop_id'] = 39
    test.loc[test['shop_id'] == 11,'shop_id'] = 10
    test.loc[test['shop_id'] == 57,'shop_id'] = 0
    test.loc[test['shop_id'] == 58,'shop_id'] = 1
    test.loc[test['shop_id'] == 40,'shop_id'] = 39

    return sales_train, test


def create_testlike_train(sales_train, test=None):
    matrix = []
    cols = ['date_block_num','shop_id','item_id']
    for i in sales_train.date_block_num.unique():
        x = itertools.product([i],
            sales_train.loc[sales_train.date_block_num == i].shop_id.unique(),
            sales_train.loc[sales_train.date_block_num == i].item_id.unique())
        matrix.append(np.array(list(x)))
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

    # Aggregate item_id / shop_id item_cnts at the month level
    sales_train_grouped = sales_train.groupby(["date_block_num", "shop_id", "item_id"]).agg(item_cnt_month=pd.NamedAgg(column="item_cnt_day", aggfunc="sum"))
    matrix = matrix.merge(sales_train_grouped, how="left", on=["date_block_num", "shop_id", "item_id"])

    # Add test data
    if test is not None:
        test["date_block_num"] = 34
        test["date_block_num"] = test["date_block_num"].astype(np.int8)
        test["shop_id"] = test.shop_id.astype(np.int8)
        test["item_id"] = test.item_id.astype(np.int16)
        test = test.drop(columns="ID")
        matrix = pd.concat([matrix, test[["date_block_num", "shop_id", "item_id"]]])

    # Fill empty with 0
    matrix.item_cnt_month = matrix.item_cnt_month.fillna(0)

    # change data type
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix['item_cnt_month'] = matrix['item_cnt_month'].astype(np.float16)
    
    return matrix


def add_item_category(matrix, items, item_categories):
    items['item_category_id'] = items['item_category_id'].astype(np.int8)
    matrix = matrix.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')

    # type_category
    type_category = item_categories["item_category_name"].str.split("-").map(lambda x: x[0])
    item_categories["type_category"] = type_category.str.split("(").map(lambda x: x[0])
    item_categories["type_code"] = item_categories["type_category"].factorize()[0].astype(np.int8)
    matrix = matrix.merge(item_categories[['item_category_id', "type_code"]], on='item_category_id', how='left')

    # subtype_category
    item_categories['split'] = item_categories['item_category_name'].str.split('-')
    item_categories['subtype_category'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip()) # if subtype is nan then type
    item_categories["subtype_code"] = item_categories["subtype_category"].factorize()[0].astype(np.int8)
    matrix = matrix.merge(item_categories[['item_category_id', "subtype_code"]], on='item_category_id', how='left')

    return matrix


def add_city_codes(matrix, shops):
    shops.loc[
        shops.shop_name == 'Сергиев Посад ТЦ "7Я"', "shop_name"
    ] = 'СергиевПосад ТЦ "7Я"'
    shops["city"] = shops["shop_name"].str.split(" ").map(lambda x: x[0])
    shops.loc[shops.city == "!Якутск", "city"] = "Якутск"
    shops["city_code"] = shops["city"].factorize()[0].astype(np.int8)
    shop_labels = shops[["shop_id", "city_code"]]
    matrix = matrix.merge(shop_labels, on='shop_id', how='left')

    return matrix


def add_sale_feature(matrix):
    first_item_block = matrix.groupby(['item_id'])['date_block_num'].min().reset_index()
    first_item_block['item_first_interaction'] = 1

    first_shop_item_buy_block = matrix[matrix['item_cnt_month'] > 0].groupby(['shop_id', 'item_id'])['date_block_num'].min().reset_index()
    first_shop_item_buy_block['first_date_block_num'] = first_shop_item_buy_block['date_block_num']

    matrix = pd.merge(matrix, first_item_block[['item_id', 'date_block_num', 'item_first_interaction']], on=['item_id', 'date_block_num'], how='left')
    matrix = pd.merge(matrix, first_shop_item_buy_block[['item_id', 'shop_id', 'first_date_block_num']], on=['item_id', 'shop_id'], how='left')

    matrix['first_date_block_num'].fillna(100, inplace=True)
    matrix['shop_item_sold_before'] = (matrix['first_date_block_num'] < matrix['date_block_num']).astype('int8')
    matrix.drop(['first_date_block_num'], axis=1, inplace=True)
    
    matrix['item_first_interaction'].fillna(0, inplace=True)
    matrix['shop_item_sold_before'].fillna(0, inplace=True)
    matrix['item_first_interaction'] = matrix['item_first_interaction'].astype('int8')  
    matrix['shop_item_sold_before'] = matrix['shop_item_sold_before'].astype('int8') 

    return matrix


def add_lag_feature(matrix, lags, col):
    tmp = matrix[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        matrix = pd.merge(matrix, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        matrix[col+'_lag_'+str(i)] = matrix[col+'_lag_'+str(i)].astype('float16')
        
    return matrix


def add_price_feature(matrix):
    index_cols = ['shop_id', 'item_id', 'date_block_num']
    group = sales_train.groupby(index_cols)['item_price'].mean().reset_index().rename(columns={"item_price": "avg_shop_price"}, errors="raise")
    matrix = pd.merge(matrix, group, on=index_cols, how='left')
    matrix['avg_shop_price'] = (matrix['avg_shop_price'].fillna(0).astype(np.float16))

    index_cols = ['item_id', 'date_block_num']
    group = sales_train.groupby(['date_block_num','item_id'])['item_price'].mean().reset_index().rename(columns={"item_price": "avg_item_price"}, errors="raise")
    matrix = pd.merge(matrix, group, on=index_cols, how='left')
    matrix['avg_item_price'] = (matrix['avg_item_price'].fillna(0).astype(np.float16))

    matrix['item_shop_price_avg'] = (matrix['avg_shop_price'] - matrix['avg_item_price']) / matrix['avg_item_price']
    matrix['item_shop_price_avg'].fillna(0, inplace=True)

    matrix = add_lag_feature(matrix, [1], 'item_shop_price_avg')
    matrix.drop(['avg_shop_price', 'avg_item_price', 'item_shop_price_avg'], axis=1, inplace=True)

    return matrix


def add_encoding_feature(matrix):
    #Add target encoding for items
    item_id_target_mean = matrix.groupby(['date_block_num','item_id'])['item_cnt_month'].mean().reset_index().rename(columns={"item_cnt_month": "item_target_enc"}, errors="raise")
    matrix = pd.merge(matrix, item_id_target_mean, on=['date_block_num','item_id'], how='left')
    matrix['item_target_enc'] = (matrix['item_target_enc'].fillna(0).astype(np.float16))
    matrix = add_lag_feature(matrix, [1], 'item_target_enc')
    matrix.drop(['item_target_enc'], axis=1, inplace=True)

    #Add target encoding for item/city
    item_id_target_mean = matrix.groupby(['date_block_num','item_id', 'city_code'])['item_cnt_month'].mean().reset_index().rename(columns={
        "item_cnt_month": "item_loc_target_enc"}, errors="raise")
    matrix = pd.merge(matrix, item_id_target_mean, on=['date_block_num','item_id', 'city_code'], how='left')
    matrix['item_loc_target_enc'] = (matrix['item_loc_target_enc'].fillna(0).astype(np.float16))
    matrix = add_lag_feature(matrix, [1], 'item_loc_target_enc')
    matrix.drop(['item_loc_target_enc'], axis=1, inplace=True)

    #For new items add avg category sales
    item_id_target_mean = matrix[matrix['item_first_interaction'] == 1].groupby(['date_block_num','type_code'])['item_cnt_month'].mean().reset_index().rename(columns={
        "item_cnt_month": "new_item_cat_avg"}, errors="raise")
    matrix = pd.merge(matrix, item_id_target_mean, on=['date_block_num','type_code'], how='left')
    matrix['new_item_cat_avg'] = (matrix['new_item_cat_avg'].fillna(0).astype(np.float16))
    matrix = add_lag_feature(matrix, [1], 'new_item_cat_avg')
    matrix.drop(['new_item_cat_avg'], axis=1, inplace=True)

    return matrix


def add_lag_feature_adv(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)+'_adv']
        shifted['date_block_num'] += i
        shifted['item_id'] -= 1
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        df[col+'_lag_'+str(i)+'_adv'] = df[col+'_lag_'+str(i)+'_adv'].astype('float16')
    return df


def postprocessing(matrix):
    matrix.fillna(0, inplace=True)
    matrix = matrix[(matrix['date_block_num'] > 2)]
    matrix.head()

    return matrix

def save_dataset(matrix):
    matrix.drop(['ID'], axis=1, inplace=True, errors='ignore')
    matrix.to_pickle('dataset.pkl')


def all_feaures(sales_train, test, item_categories, items, shops):
    print('data_cleaning')
    sales_train, test = data_cleaning(sales_train, test)

    print('create_testlike_train')
    matrix = create_testlike_train(sales_train, test)

    print('add_item_category')
    matrix = add_item_category(matrix, items, item_categories)

    print('add_city_codes')
    matrix = add_city_codes(matrix, shops)

    print('add_sale_feature')
    matrix = add_sale_feature(matrix)

    # Add sales lags for last 3 months
    print('add_lag_feature')
    matrix = add_lag_feature(matrix, [1, 2, 3], 'item_cnt_month')

    # Add avg shop/item price lags
    print('add_price_feature')
    matrix = add_price_feature(matrix)

    print('add_encoding_feature')
    matrix = add_encoding_feature(matrix)

    print('add_lag_feature_adv')
    matrix = add_lag_feature_adv(matrix, [1], 'item_cnt_month')

    matrix = postprocessing(matrix)

    save_dataset(matrix)


def load_dataset():
    # load dataset
    data = pd.read_pickle("dataset.pkl")

    # drop features to reduce overfitting
    data = data.drop(columns = ["shop_id", "item_id"])

    X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    del data

    return X_train, Y_train, X_valid, Y_valid, X_test


def train(X_train, Y_train, X_valid, Y_valid):
    # model parameter
    model = XGBRegressor(
        max_depth=8,
        n_estimators=1000,
        min_child_weight=300, 
        colsample_bytree=0.8, 
        subsample=0.8, 
        eta=0.3,    
        seed=42)

    # train
    model.fit(
        X_train, 
        Y_train, 
        eval_metric="rmse", 
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
        verbose=True, 
        early_stopping_rounds = 10)

    return model

def predict(model, X_test):
    Y_test = model.predict(X_test).clip(0, 20)

    submission = pd.DataFrame({
        "ID": test.index, 
        "item_cnt_month": Y_test
    })

    return submission


def save_submission(submission, output):
    submission.to_csv(output, index=False)


if __name__ == '__main__':
    s = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--sales_train', 
                        default='sales_train.csv',
                        help= 'input sales_train file')
    parser.add_argument('--test', 
                        default='test.csv',
                        help= 'input test file')
    parser.add_argument('--item_categories', 
                        default='item_categories.csv',
                        help= 'input item_categories file')                        
    parser.add_argument('--items', 
                        default='items.csv',
                        help= 'input items file')
    parser.add_argument('--shops', 
                        default='shops.csv',
                        help= 'input shops file')
    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # load data
    sales_train = pd.read_csv(args.sales_train)
    test = pd.read_csv(args.test)
    item_categories = pd.read_csv(args.item_categories)
    items = pd.read_csv(args.items)
    shops = pd.read_csv(args.shops)

    # make features
    all_feaures(sales_train, test, item_categories, items, shops)

    i = time.time()
    print(i-s)

    # load dataset
    X_train, Y_train, X_valid, Y_valid, X_test = load_dataset()

    # train
    print('train')
    model = train(X_train, Y_train, X_valid, Y_valid)

    # predict
    submission = predict(model, X_test)
    
    # save submission
    save_submission(submission, args.output)
    print('done')

    e = time.time()
    print(e-i)