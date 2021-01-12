import os
import pickle
import json
import numpy
import joblib
from utils import reduce_mem, unique_count


def preprocess_data(df):
    """ Perform brief data preprocessing for the incoming dataset object """

    # categorical feature unique count
    categorical_cols = [
        "slot_id",
        "adv_id",
        "adv_prim_id",
        "creat_type_cd",
        "inter_type_cd",
        "age",
        "city",
        "uid",
        "dev_id",
        "task_id",
    ]
    df = unique_count(df, categorical_cols)
    df = reduce_mem(df)

    # drop engineered features
    drop_fea = ["pt_d", "communication_onlinerate", "uid"]
    df.drop(columns=drop_fea, inplace=True)

    return df


def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = pd.DataFrame(json.loads(raw_data))
        data = preprocess_data(data)
        result = model.predict_proba(data)[:, 1]
        # you can return any data type as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
