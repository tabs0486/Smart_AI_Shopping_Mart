import pandas as pd


def load_products():
    return pd.read_csv("data/products.csv")


def search_products(
    category=None,
    product=None,
    maximum_budget=None,
):
    df = load_products()

    if category:
        df = df[
            df["category"].str.contains(
                category,
                case=False,
                na=False
            )
        ]

    if product:
        df = df[
            df["name"].str.contains(
                product,
                case=False,
                na=False
            )
        ]

    if maximum_budget:
        df = df[
            df["price"] <= maximum_budget
        ]

    return df
