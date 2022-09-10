def create_target_variable(df):
    """
    Creates the target variable: Hourly Call Volume (Number of Calls per hour)
    :param df: pd.Dataframe, containing the preprocessed data
    :return: df (with target variable)
    """
    y_m_d= df.apply(lambda x: (int(x["Year"]), int(x["Month"]),int(x["Day"]), int(x["Hour"])), axis=1)
    hourly_call_volume = df.groupby(by=["Year", "Month", "Day", "Hour"]).count()["Latitude"].to_dict()
    df["Hourly Call Volume"] = y_m_d.map(hourly_call_volume)
    return df

def add_daily_call_volume_feat(df):
    """
    Adds a new feature: Daily Call Volume (Number of calls per day)
    :param df: pd.Dataframe, containing the preprocessed data
    :return: df (with new feature)
    """
    m_d = df.apply(lambda x: (int(x["Year"]), int(x["Month"]),int(x["Day"])), axis=1)
    avg_daily_call_volumes = df.groupby(by=["Year", "Month", "Day"])["Hour"].count().to_dict() #.apply(lambda x: x/df["Year"].unique().size)
    df["Daily Call Volume"] = m_d.map(avg_daily_call_volumes)
    return df

def get_season(in_datetime):
    """
    Returns season of datetime object
    :param in_datetime: datetime.datetime
    :return: season (str): winter, spring, summer or fall
    """
    Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
    seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
               ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
               ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
               ('fall', (date(Y,  9, 23),  date(Y, 12, 20))),
               ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

    assert isinstance(in_datetime, datetime), "Not a datetime object!"
    in_datetime = in_datetime.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= in_datetime <= end)

def create_features(df):
    """
    Creates several new features and returns an Easydict containing the train and test data
    :param df: pd.Dataframe, containing preprocessed data
    :return: EasyDict containing both train and test data
    """
    df["Season"]= df["Datetime"].apply(lambda x: get_season(x))
    df["Weekday"]= df["Datetime"].apply(lambda x: datetime.weekday(x))
    df = add_daily_call_volume_feat(df)

    df = create_target_variable(df)

    df.sort_values(by='Datetime', inplace=True) # Sort by Date
    df.drop(["Datetime"], axis=1, inplace=True) # All necessary information already exported to other columns

    # We want to predict the hourly call volume of 2019
    df_train = df[df["Year"] < 2019]
    df_test = df[df["Year"] == 2019]

    target_train = df_train.pop("Hourly Call Volume")
    target_test = df_test.pop("Hourly Call Volume")

    return EasyDict(X_train=df_train,y_train=target_train, X_test=df_test, y_test=target_test)
