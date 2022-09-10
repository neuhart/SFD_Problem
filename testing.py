def test_tree(x_train, y_train, x_test, y_test):
    """
    Evaluates the
    :param x_train: pd.Dataframe, containing the features of the training data
    :param y_train: pd.Series, containing the target variable of the training data
    :param x_test: pd.Dataframe, containing the features of the test data
    :param y_test: pd.Series, containing the target variable of the test data
    :return:
    """

    pipe = joblib.load("./dec_tree_pipe.joblib")

    train_pred = pipe.predict(x_train)
    train_loss = mean_squared_error(y_train, train_pred)
    train_R2_score = pipe.score(x_train, y_train)
    print('Training MSE Loss: {} \n Test R2 score {}'.format(train_loss, train_R2_score))

    test_pred = pipe.predict(x_test)
    test_loss = mean_squared_error(y_test, test_pred)
    test_R2_score  = pipe.score(x_test, y_test)
    print('Test MSE Loss: {} \n Test R2 score {}'.format(test_loss, test_R2_score))
    print(type(test_pred))
    return test_pred
