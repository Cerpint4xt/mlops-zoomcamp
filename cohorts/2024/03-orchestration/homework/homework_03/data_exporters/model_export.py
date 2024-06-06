import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

EXPERIMENT_NAME = "ny_taxi_experiments"

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
#mlflow.set_experiment("nyc-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(df, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    categorical = ['PULocationID', 'DOLocationID']
    #numerical = ['trip_distance']
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = df[target].values

    with mlflow.start_run():

        mlflow.set_tag("developer", "raul")
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")


