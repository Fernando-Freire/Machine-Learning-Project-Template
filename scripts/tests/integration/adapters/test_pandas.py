import pandas as pd

from scripts.adapters.pandas import Pandas

storage_options = {
    "key": "minio",
    "secret": "minio123",
    "client_kwargs": {"endpoint_url": "http://minio:9000"},
}


def test_get_classes():

    df_expected = pd.DataFrame(
        {
            "product_id": ["1", "2", "3", "4", "5"],
            "title": ["maça", "chuchu", "agrião", "pera", "tomate"],
            "concatenated_tags": [
                "red gala fuji verde",
                "verde mucho maduro",
                "amargo verde escuro",
                "amarela doce macia",
                "vermelho grande cereja",
            ],
            "category": ["fruta", "legume", "verdura", "fruta", "fruta"],
        }
    )
    # print(df_expected)
    # print("carammmaabaaaaaaS")
    path = "s3://data/" + "test_integration_2.csv"
    df_expected.to_csv(path, storage_options=storage_options)

    # pandas = Pandas(storage_options=storage_options)

    # df_received = pandas.get_data(
    #     base_path="s3://data/", file_name="test_integration_2.csv"
    # )

    df_sent = Pandas._read_data_from_s3(storage_options=storage_options, filepath=path)

    print(df_sent)
    categories = Pandas.get_classes(storage_options=storage_options, filepath=path)

    print(categories)

    assert len(categories) == 3
    assert "fruta" in categories
    assert "verdura" in categories
    assert "legume" in categories


def test_get_data():

    df_expected = pd.DataFrame(
        {
            "product_id": ["1", "2", "3", "4", "5"],
            "title": ["maça", "chuchu", "agrião", "pera", "tomate"],
            "concatenated_tags": [
                "red gala fuji verde",
                "verde mucho maduro",
                "amargo verde escuro",
                "amarela doce macia",
                "vermelho grande cereja",
            ],
            "category": ["fruta", "legume", "verdura", "fruta", "fruta"],
        }
    )

    path = "s3://data/" + "test_integration_3.csv"
    df_expected.to_csv(path, storage_options=storage_options)

    result = Pandas._get_data(storage_options=storage_options, filepath=path)

    assert len(result.columns) == 5
    assert "fruta" in result.columns
    assert "verdura" in result.columns
    assert "legume" in result.columns
