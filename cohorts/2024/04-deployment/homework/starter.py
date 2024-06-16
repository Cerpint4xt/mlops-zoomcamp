#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd

#year = 2024
#month = 3
#taxi_type = 'yellow'



def generate_ride_id(df, year, month):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df

def read_dataframe(filename:str, year, month):
    df = pd.read_parquet(filename)
    categorical = ['PULocationID', 'DOLocationID']
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df = generate_ride_id(df, year, month)
    return df

def prepare_dictionaries(df:pd.DataFrame):

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    return dicts

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def save_results(df, y_pred, output_file):
   df_result = pd.DataFrame()
   df_result['ride_id'] = df['ride_id']
   df_result['predicted_duration'] = y_pred
   df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
    )


def apply_model(input_file, output_file, year, month):
    print('Reading data from {input_file}...')
    df = read_dataframe(input_file, year, month)
    dicts = prepare_dictionaries(df)
    print('loading the model from source...')
    dv, model = load_model()
    X_val = dv.transform(dicts)
    print('applying the model...')
    y_pred = model.predict(X_val)
    print('y_pred mean:', y_pred.mean())
    save_results(df, y_pred, output_file)
    return output_file


def run():
    taxi_type = sys.argv[1]
    year = int(sys.argv[2])
    month = int(sys.argv[3])
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    apply_model(input_file=input_file, output_file=output_file, year=year, month=month)

if __name__ == '__main__':
    run()














