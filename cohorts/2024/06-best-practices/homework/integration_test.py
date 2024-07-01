import os
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# Create the dataframe for January 2023
data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

# S3 endpoint URL for Localstack
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
bucket_name = "nyc-duration"
input_file = f's3://{bucket_name}/in/2023-01.parquet'

options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

# Save the dataframe to S3 (Localstack)
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

print(f"Input file saved to {input_file}")

# Run the batch script for January 2023
os.system('python batch.py 2023 01')

# Read the output data from S3 (Localstack)
output_file = f's3://{bucket_name}/in/2023-01.parquet'
df_output = pd.read_parquet(output_file, storage_options=options)

# Verify the output
print("Output data:")
print(df_output)
