import pandas as pd
import numpy as np

# Calculating stats utilites
def describe_data(data):
    print("-------------Size-------------")
    print(data.shape)

    print("\n\n---------Metadata Info--------")
    print(data.info())

    print("\n\n----------Sample Data---------")
    print(data.head())

    print("\n\n------------Summary-----------")
    print(data.describe().T)

