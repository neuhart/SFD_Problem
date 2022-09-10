def preprocessing(raw_df):
    """
    Prepocesses the data and returns dataframe
    :param raw_df: pd.Dataframe
    :return: df(pd.Dataframe)
    """
    # Divide date into Years,Months,Days,Hours
    raw_df["Datetime"]= raw_df["Datetime"].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))
    raw_df["Year"] = raw_df["Datetime"].apply(lambda x: int(x.year))
    raw_df = raw_df[raw_df["Year"] >= 2014]
    raw_df["Month"] = raw_df["Datetime"].apply(lambda x: int(x.month))
    raw_df["Day"] = raw_df["Datetime"].apply(lambda x: int(x.day))
    raw_df["Hour"] =raw_df["Datetime"].apply(lambda x: int(x.hour))

    raw_df.dropna(inplace=True) # drop null values

    df = raw_df.copy()
    # drop features we won't use
    df.drop(["Address", "Type", "Report Location", "Incident Number"], axis=1, inplace=True)
    return df
