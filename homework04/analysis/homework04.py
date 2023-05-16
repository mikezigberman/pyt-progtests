"""
In this homework, we will use well-known Titanic dataset which contains 
information about passengers of Titanic. The dataset consists of personal 
information about each passenger and indicator whether the passenger 
survived. We will use this data to analyse passenger list and their chance for
survival.

The provided dataset contains the following attributes:
 'Age' - age in years,
 'Fare' - fare ticked price,
 'Name' - passenger name,
 'Parch' - # of parents/children of a person on board,
 'PassengerId' - identifier,
 'Pclass' - travelling class, 1 = 1. class, 2 = 2. class, 3 = 3. class,
 'Sex' - sex,
 'SibSp' - # siblings/spouses on board,
 'Survived' - 0 = died, 1 = survived,
 'Embarked' - boarding port C = Cherbourg, Q = Queenstown, S = Southampton,
'Cabin' - cabin number
 'Ticket' - ticket number
"""
import numpy
import pandas as pd


def load_dataset(train_file_path: str, test_file_path: str) -> pd.DataFrame:
    """
    Write a function which loads CSV from two files to pandas DataFrame and
    performs several data processing steps. Use data provided in `data`
    directory for testing ('data/train.csv' as input parameter
    `train_file_path`, and 'data/test.csv'  as `test_file_path`). 

    Add column name "Label" to each DataFrame. The column should contain value "Train"
    for data from `train_file_path` and "Test" from test_file_path.

    Perform following operations with DataFrames (keep the order of the
    operations):
        1. Concatenate both DataFrames.
        2. Remove columns  "Ticket", "Embarked", "Cabin" from created DataFrame.
        3. Set the index to unique numbers from zero to the number of rows.

    The return value of the function is processed DataFrame.
    """

    # read data from file
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    # add column "Label"
    train_df["Label"] = 'Train'
    test_df["Label"] = 'Test'

    # 1. Concatenate both DataFrames.
    data = pd.concat([train_df, test_df])

    # 2. Remove columns  "Ticket", "Embarked", "Cabin" from created DataFrame.
    data.drop(["Ticket", "Embarked", "Cabin"], axis=1, inplace=True)

    # 3. Set the index to unique numbers from zero to the number of rows.
    data.reset_index(inplace=True)

    data.drop(["index"], axis=1, inplace=True)

    return data


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    When working and analysing data, one often needs to deal with missing
    values. For example, some passengers did not fill information about
    family members. In that case, one needs to be aware of it as it may
    introduce bias to the data.

    Write a function which determines the number of missing values in given
    DataFrame. 

    The function should output a new DataFrame. The new DataFrame
    should be indexed by columns of original DataFrame. Columns of returned
    DataFrame will be (keep the order of the columns):
        1. "Total" - contains the number of missing values
        2. "Percent" - contains the percentage of missing values with regard to all
        rows of given DataFrame.

    Sort the resulting DataFrame based on the number of missing values from
    largest to smallest.

    Example of output:

               |  Total  |  Percent
    "Column1"  |   34.5  |    76.54321
    "Column2"  |   0     |    0
    """

    # create empty dataframe
    df_new = pd.DataFrame()

    # fill new df indexes as columns names of given df
    df_new["index"] = df.columns
    df_new.set_index('index', inplace=True)
    df_new.index.set_names([""], inplace=True)

    # add columns
    df_new["Total"] = 0
    df_new["Percent"] = 0

    # fill columns Total and Percent
    for i in range(0, len(df.columns)):
        df_new.iloc[i, 0] = df.iloc[:, i].isna().sum()

        df_new.iloc[i, 1] = (100 * df.iloc[:, i].isna().sum())/len(df.index)

    # sort df
    df_new.sort_values(by=['Total'], ascending=False, inplace=True)

    return df_new


def substitute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    One way how to handle missing data is to substitute missing values with
    some statistic of other rows. We will use this method for two columns:
        1. "Age" - fill missing values with the mean of other rows.
        2. "Fare" - fill missing values with the lowest price of ~$15 (we
        suppose that the majority of unregistered tickets were the cheapest
        ones).

    Do not to modify given DataFrame but create a copy of it.
    """

    # Do not to modify given DataFrame but create a copy of it.
    df_copy = df.copy()

    # 1. "Age" - fill missing values with the mean of other rows.
    df_copy["Age"].fillna(value=df_copy["Age"].mean(), inplace=True)

    # 2. "Fare" - fill missing values with the lowest price of ~$15 (we
    # suppose that the majority of unregistered tickets were the cheapest
    # ones).
    df_copy["Fare"].fillna(value=15, inplace=True)

    return df_copy


def get_correlation(df: pd.DataFrame) -> float:
    """
    We want to know whether there is a relationship between the age of a
    passenger and fare ticket price (e.g. younger children have cheaper
    tickets). We will use Pearson correlation coefficient to quantify linear
    relationship between columns "Age" and "Fare".
    The result will be returned as one number.

    Pearson correlation coefficient quantifies linear relationship between
    two random variables. Correlation ranges from -1 to 1. Value around zero
    indicates no linear relationship, -1 indicates strong negative
    relationship, 1 indicates strong relationship.
    """

    return df["Age"].corr(df["Fare"])


def get_survived_per_class(df: pd.DataFrame,
                           group_by_column_name: str) -> pd.Series:
    """
    We want to know how big was the chance of survival for different groups of
    passengers (e.g. for different sexes, classes, etc.). 

    Write a function
    that estimates that. The input of the function is a DataFrame with data
    and name of column (group_by_column_name) which holds group information.
    To increase readability of the Ыresult sort values from the highest chance of
    survival to lowest and round the resulting values to 2 decimal places.
    Return result as pandas Series.

    Example:

    get_survived_per_class(df, "Sex")

                  Survived
    Female     |      0.82
    Male       |      0.32

    """

    # Create DataFrame for filling
    series = pd.DataFrame(columns=['index', "Survived"])
    series['index'] = df[group_by_column_name].unique()
    series['Survived'] = series['Survived'].astype("float")

    data = df.copy()
    data = data[data["Survived"].notna()]
    data = data[data[group_by_column_name].notna()]

    # sort values for right access
#     series.sort_values(by=['index'])

    # filling DataFrame
    row_count = len(data[group_by_column_name])

    index_series = list(data[group_by_column_name].unique())
#     index_series.sort() # sort values for right access

    index_int = 0

    # we could use group by function!!!
    for i in index_series:
        ones = 0
        count = 0

        for j in range(0, row_count):
            if i == data[group_by_column_name][j]:
                count += 1

                if df["Survived"][j] == 1:
                    ones += 1

        series.iloc[index_int, 1] = round(ones / count, 2)

        index_int += 1

    ###
#     display(series)

    # convert DataFrame to Series
    series.set_index('index', inplace=True)
    series.index.set_names([""], inplace=True)

    series = series.iloc[:, 0]

    # sort Series
    series = series.sort_values(ascending=False)

    return series


def get_outliers(df: pd.DataFrame) -> (int, pd.DataFrame):
    """
    We want to explore fare ticket prices. An important part of such
    exploration is exploration of outliers. An outlier may indicate an error
    in the data (somebody entered price incorrectly) or some special group of
    passengers.

    We will use the IQR method for the identification of outliers. IQR method
    considers an outlier any point which does not fulfil:
        Q1 - 1.5*IQR < point_value < Q3 + 1.5*IQR,
    where Q1 and Q3 are the first and the third quartiles respectively
    calculated from all points in data. IQR is the inter-quartile range
    calculate as the difference between Q3 and Q1:
        IQR = Q3 - Q1.

    Return tuple with the number of outliers and all passengers with outlier
    fare ticket price.
    """

    q1 = df['Fare'].quantile(0.25)

    q3 = df['Fare'].quantile(0.75)

    IQR = q3-q1

    outliers = df[((df['Fare'] < (q1-1.5*IQR)) | (df['Fare'] > (q3+1.5*IQR)))]

    outliers = outliers[outliers['Fare'].notna()]

    return (len(outliers.iloc[:, 0]), outliers)


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    To analyse data and use them for modeling, it may be convenient to create
    a new columns (features). These new features are usually created
    transformation of original values. For example, if we want to compare
    survivals from Titanic and SS Eastland we will want to scale fare prices
    to the same values for each ship as travelling on Titanic was more
    expensive.

    Create 3 new variables:
        1. "Fare_scaled" - scale "Fare" columns to have zero mean and standard
       deviation equal one.
        2. "Age_log" - is natural logarithm of attribute "Age" (differences
        between age of children are magnified in comparison to adults).
        3. "Sex" -  Replace string values with numerical ones, where "male"
        will be replaced with 0 and "female" with 1. The resulting values
        should have type `int`.

    Do not modify original DataFrame.
    """
    # Do not modify original DataFrame.
    data = df.copy()

    # 1. "Fare_scaled" - scale "Fare" columns to have zero mean and standard
    # deviation equal one.
    # np.std - numpy standard deviation
    data["Fare_scaled"] = (data.Fare-data.Fare.mean())/numpy.std(data.Fare)

    # 2. "Age_log" - is natural logarithm of attribute "Age" (differences
    # between age of children are magnified in comparison to adults).
    data["Age_log"] = numpy.log(data["Age"])

    # 3. "Sex" -  Replace string values with numerical ones, where "male"
    # will be replaced with 0 and "female" with 1. The resulting values
    # should have type `int`.
    data["Sex"].replace('female', int(1), inplace=True)
    data["Sex"].replace('male', int(0), inplace=True)

    return data


def determine_survival(df: pd.DataFrame, n_interval: int, age: float,
                       sex: str) -> float:
    """
    Determine the probability of survival of a person specified by age and sex.

    Missing values in column "Age" replace with mean value. In order to
    moderate significance of the estimated probability, divide "Age" to
    specified number of intervals and calculate probability from given
    interval. For example if we have values in "Age" column [2, 13, 18, 25] and
    we want 2 intervals, result should be:

    0    (1.977, 13.5]
    1     (13.5, 25.0]

    With division based on "Sex", the categorization should be:

       "AgeInterval" | "Sex"       |   "Survival Probability"
       (1.977, 13.5] | "male"      |            0.21
       (1.977, 13.5] | "female"    |            0.28
       (13.5, 25.0]  | "male"      |            0.10
       (13.5, 25.0]  | "female"    |            0.15

    Output of determine_survival(df, n_interval=2, age = 5, sex = "male")
    should be 0.21. If there is no passenger for some group, return numpy
    NA value.
    """
    # Replace missing values in column "Age" with the mean value.
    df.Age.fillna(df.Age.mean(), inplace=True)

    # Select columns "Survived", "Sex", and "Age".
    df = df[["Survived", "Sex", "Age"]]

    # Remove rows where "Survived" is missing and convert "Survived" to integer.
    df = df[df.Survived.notna()]
    df.Survived = df.Survived.astype(int)

    # Select only rows with the specified "sex".
    df = df[df.Sex == sex]

    # Select columns "Survived" and "Age".
    df = df[["Survived", "Age"]]
    # Divide "Age" into "n_interval" intervals and create a new column "Age_group".
    df["Age_group"] = pd.cut(x=df.Age, bins=n_interval, ordered=True)
    # Select the row where "Age_group" contains the specified "age".
    interval = df[df["Age_group"].apply(lambda x: age in x)]
    # If no row contains the specified "age", return numpy.NAN.
    if interval.empty:
        return numpy.nan

    # Select the rows where "Age_group" matches the selected "interval".
    df = df[df["Age_group"] == interval.iloc[0]["Age_group"]]

    # Calculate the survival probability.
    ones = df["Survived"].sum()
    count = df["Survived"].count()
    return float(ones / count) if count != 0 else numpy.nan
