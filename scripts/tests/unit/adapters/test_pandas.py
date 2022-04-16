import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.adapters.pandas import Pandas

storage_options = {
    "key": "minio",
    "secret": "minio123",
    "client_kwargs": {"endpoint_url": "http://minio:9000"},
}

df = pd.DataFrame(
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


def test_basic_preprocessing():
    df2 = pd.DataFrame(
        {
            "product_id": ["1", "2", "3", "4", "5", "5"],
            "title": ["maça", "chuchu", "agrião", "pera", "tomate", "tomate"],
            "concatenated_tags": [
                "red gala fuji verde",
                "verde mucho maduro",
                "amargo verde escuro",
                "amarela doce macia",
                "vermelho grande cereja",
                "vermelho grande cereja",
            ],
            "category": ["fruta", "legume", "verdura", "fruta", "fruta", "fruta"],
        }
    )

    df_expected = df.copy()
    print(df_expected)
    df_expected = df_expected.drop_duplicates("product_id")
    print(df_expected)
    df_expected["text"] = df_expected["concatenated_tags"] + " " + df_expected["title"]
    print(df_expected)
    df_expected = df_expected[["text", "category"]]
    print(df_expected)

    pandas = Pandas()
    df_received = pandas._basic_preprocessing(df=df2)
    assert_frame_equal(df_received, df_expected)


def test_one_hot_encoding():

    pandas = Pandas()
    df_expected = df.copy()
    df_expected = pd.concat([df_expected, pd.get_dummies(df_expected.category)], axis=1)

    df_received = pandas._one_hot_encoding(df=df)

    assert_frame_equal(df_expected, df_received)


# def test_split_data_set():
#    assert(True)

# def test_get_classes():

#     raw = RawData(df=df)
#     classes = Pandas.get_classes(raw)

#     assert "fruta" in classes
#     assert "legume" in classes
#     assert "verdura" in classes
#     assert len(classes) == 3
