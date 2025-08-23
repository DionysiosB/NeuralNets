import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv("https://raw.githubusercontent.com/ageron/data/refs/heads/main/housing/housing.csv")

df.head()
df.info()
df.describe()
df.hist(bins=50)
print(df.columns)

df.iloc[:5][["total_rooms", "total_bedrooms"]] ##Get a subset of the dataframe

X = df.drop(columns=["median_house_value"])
y = df[["median_house_value"]]

print(y.isna().sum()) ## Check if there are any NaNs in the Ys
X.isna().sum(axis=0)

#X.dropna(subset=["total_bedrooms"])

from sklearn.impute import SimpleImputer
medianImputer = SimpleImputer(strategy="median")
Xm = X.copy()
Xm = Xm.select_dtypes(include=[np.number])
medianImputer.fit(Xm)
Xm = pd.DataFrame(medianImputer.transform(Xm), columns = Xm.columns)
Xm.head()


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X[["ocean_proximity"]])
dfonehot = pd.DataFrame(encoder.transform(X[["ocean_proximity"]]), columns=encoder.categories_)
dfonehot.head()


from sklearn.preprocessing import MinMaxScaler, StandardScaler
mms = MinMaxScaler(feature_range=(-1, 1))
ss  = StandardScaler()
Xm = X.select_dtypes(include=[np.number])
mms.fit_transform(Xm)
#ss.fit_transform(Xm)


from sklearn.pipeline import Pipeline, make_pipeline
num_pipeline = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", MinMaxScaler(feature_range=(-1, 1)))
    ]
)
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
Xm = X.select_dtypes(include=[np.number])
Xm = pd.DataFrame(num_pipeline.fit_transform(Xm), columns=Xm.columns)
Xm.head()


from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

Xm = preprocessing.fit_transform(X)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
linreg = make_pipeline(preprocessing, model)
linreg.fit(Xtrain, ytrain)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
Xm = X.drop(columns=["ocean_proximity"])
treereg = make_pipeline(preprocessing, model)
treereg.fit(Xm, y)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
Xm = X.drop(columns=["ocean_proximity"])
forestreg = make_pipeline(preprocessing, model)
forestreg.fit(Xm, y)

from sklearn.model_selection import cross_val_score
#rmses = cross_val_score(linreg, Xm, y, cv=10)
#rmses = cross_val_score(treereg, Xm, y, cv=10)
rmses = cross_val_score(forestreg, Xm, y, cv=10)
print(rmses)
