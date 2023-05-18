

# 一个预测房价的AI
# 这是final project，已经删除了大量不必要的代码，只保留了建模和执行预测功能的代码，想看更详细的过程移步test.py


from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats


# 数据显示的最大列
pd.options.display.max_columns = 50
# 房价的URL
URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"


# --------------------------------获取资源----------------------------------------------------------
# --------------这是按照本人主机上的相对路径创建/寻找资源的，若有需要，请自行改动本地url--------------
def load_housing_data(url=URL):
    dataPath = Path("datasets/housing/housing.tgz")
    if not dataPath.is_file():
        Path("datasets/housing").mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dataPath)
        with tarfile.open(dataPath) as dataDownLoad:
            dataDownLoad.extractall(path="datasets/housing")
    # print("已获得数据：" + str(dataPath) + "，即将解压")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


housing_no_label = load_housing_data()


# -----------------------------数据准备阶段---------------------------------------
# 将收入按中位数重新分层
housing_no_label["income_cat"] = pd.cut(housing_no_label["median_income"],
                                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                        labels=[1, 2, 3, 4, 5])


# 将数据集划分为训练集和测试集（n_split参数为1，所以只划分一次，后期需要多次validate的话再说）
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing_no_label, housing_no_label["income_cat"]):
    strat_train_set = housing_no_label.loc[train_index]
    strat_test_set = housing_no_label.loc[test_index]


# 删除这一列（income_cat）
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    # set_.info()


# -------------------------------数据探索(想看数据探索移步test.py)--------------------------------------


# -----------------------------------开始构造特征矩阵（真正的数据准备阶段）---------------------------------------------
# 将要预测的值（label）和其他值分开
housing_no_label = strat_train_set.drop("median_house_value", axis=1)
housing_label = strat_train_set["median_house_value"].copy()
# 把非数值数据(ocean_proximity)分离出来
housing_num = housing_no_label.drop("ocean_proximity", axis=1)
# 文本数据
housing_cat = housing_no_label[["ocean_proximity"]]


# -----------------自己创建一个类，该类可以自己创建一个组合属性（自定义转换器）-------------------------
# 下面4个值代表了本项目导入的数据集（house）中的几个属性的位置（具体什么属性看名字就明白了）
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]  # 将X的第3列和第4列每行分别做除法，得到一维数组，长度为X的行数
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# -----------------------------------------------------------------------------------------


# 流水线（把前面的操作按顺序整合在一起）
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

# 自定义转换器，可以按不同的方式来同时处理不同的数据。这里用之前的流水线处理数值数据，用独热编码处理分类数据
num_attribs = list(housing_num)
cat_attribs = list(housing_cat)
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
feature_matrix = full_pipeline.fit_transform(housing_no_label)  # 转换成特征矩阵
print(feature_matrix[:5])

# -----------------------------------------选择模型(具体怎么选，不在这里看)-------------------------------------------


# -----------------------------------微调参数-----------------------------------------
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(feature_matrix, housing_label)


# 最后，将找到的最好参数应用到模型中
final_model = grid_search.best_estimator_
# 使用测试集
test_no_label = strat_test_set.drop("median_house_value", axis=1)
test_label = strat_test_set["median_house_value"].copy()
test_prepared = full_pipeline.transform(test_no_label)  # 别用fit或者fit_transform，你也不希望纯洁的测试集被NTR吧
final_predictions = final_model.predict(test_prepared)
# 测测准确度
final_mse = mean_squared_error(test_label, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
# 如果对之前的点估计不够自信，可以算一下置信区间
confidence = 0.95
squared_errors = (final_predictions - test_label) ** 2
X = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
print("置信区间:", X)
