def training(x_train, y_train):
    """
    Trains the Decision Tree
    :param x_train: pd.Dataframe, containing the features of the training data
    :param y_train: pd.Series, containing the target variable of the training data
    """

    numerical_cols = x_train.select_dtypes('number').columns
    categorical_cols = pd.Index(np.setdiff1d(x_train.columns, numerical_cols))
    rng = np.random.RandomState(1)
    tree = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=5), n_estimators=300, random_state=rng)

    numerical_pipe = Pipeline([
        ('scaler', StandardScaler())])

    categorical_pipe = Pipeline([(
        'encoder', OneHotEncoder(drop='first', handle_unknown='error'))])

    preprocessors = ColumnTransformer(transformers=[
        ('num', numerical_pipe, numerical_cols),
        ('cat', categorical_pipe, categorical_cols)
    ])

    pipe = Pipeline([('preprocessors', preprocessors), ('tree', tree)])

    i=100000
    pipe.fit(x_train.iloc[:i,:], y_train.iloc[:i])

    joblib.dump(pipe, "./dec_tree_pipe.joblib")
