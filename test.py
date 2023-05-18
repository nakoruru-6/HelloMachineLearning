from pathlib import Path
import pandas as pd
from pandas.plotting import scatter_matrix
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
import joblib

# 数据显示的最大列
pd.options.display.max_columns = 50
# 房价的URL
URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"


# -------------------------获取资源-----------------------------------------------

def load_housing_data(url=URL):
    dataPath = Path("datasets/housing/housing.tgz")
    if not dataPath.is_file():
        Path("datasets/housing").mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dataPath)
        with tarfile.open(dataPath) as dataDownLoad:
            dataDownLoad.extractall(path="datasets/housing")
    # print("已获得数据：" + str(dataPath) + "，即将解压")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


housing = load_housing_data()

# print("----------前十条数据----------")
# print(housing.head(10))
# print()
# print()

# print("----------后十条数据----------")
# print(housing.tail(10))
# print()
# print()

# print("----------返回指定数据----------")
# print(housing.loc[[12654,12655]])
# print()
# print()

# print("----------基本信息----------")
# housing.info()
# print()
# print()

# print("----------地理分布----------")
# print(housing['ocean_proximity'].value_counts())
# print()
# print()

# print("----------描述（我也不知道这函数干嘛用的）----------")
# print(housing.describe())
# print()
# print()

# currentPath = Path.cwd()
# homePath = Path.home()
# print("文件当前所在目录:%s\n用户主目录:%s" % (currentPath, homePath))


# makePath = currentPath / 'python-100'
# makePath.mkdir()
# print("创建的目录为:%s" %(makePath))
# delPath = currentPath / 'python-100'
# delPath.rmdir()
# print("删除的目录为:%s" %(delPath))


# extra code – the next 5 lines define the default font sizes
# plt.rc('font', size=14)
# plt.rc('axes', labelsize=14, titlesize=14)
# plt.rc('legend', fontsize=14)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
#
# housing.hist(bins=50, figsize=(12, 8))
# plt.show()


# -----------------------------数据准备阶段---------------------------------------
# 将收入按中位数重新分层
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# housing.info()
# housing["income_cat"].hist()
# plt.show()

# 将数据集划分为训练集和测试集（n_split参数为1，所以只划分一次，后期需要多次validate的话再说）
# 然后看看划分出的训练集和测试集分布是否和总集一样
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print("检查划分出的训练集和测试集是否具有代表性")
# print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# 删除这一列（income_cat）
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    # set_.info()

# -------------------------------保存副本--------------------------------------
housing = strat_train_set.copy()

# -------------------------------数据探索--------------------------------------

# # 绘制色图
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.15, grid="True",
#              s=housing["population"] / 100, label="population", figsize=(10, 7),
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#              )
# # plt.legend()
# plt.show()


# 显示部分属性之间的相关性
# attributes = ["median_house_value", "median_income", "total_rooms",
#               "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()


# 显然只需看房价与收入的关系就够了
# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#              alpha=0.1)
# plt.show()


# 这才哪到哪，我们还可以尝试用如下的方式创造新的属性，这些新属性很显然是有用的，可以观察它们与房价的关系
# housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
# housing["population_per_household"] = housing["population"] / housing["households"]
# 查询各属性的相关系数
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
# housing.plot(kind="scatter", x="median_income", y="rooms_per_household", alpha=0.15)
# plt.show()
# attributes = ["median_income", "rooms_per_household", "bedrooms_per_room", "population_per_household"]
# scatter_matrix(housing[attributes], figsize=(15, 10))
# plt.show()


# -----------------------------------开始构造特征矩阵（真正的数据准备阶段）---------------------------------------------
# 将要预测的值（label）和其他值(predictor)分开
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
# 马上要填充空白的值（total_bedrooms数据有缺失），使用中位数来填充
# 首先把非数值数据(ocean_proximity)分离出来
housing_num = housing.drop("ocean_proximity", axis=1)
# 计算剩余部分的中位数，并将中位数填充至空白部分
# imputer = SimpleImputer(strategy="median")
# imputer.fit(housing_num)
# X = imputer.transform(housing_num)
# housing_transform = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
# housing_transform.info()


# 数值数据处理完了，接下来处理文本数据
housing_cat = housing[["ocean_proximity"]]
# categories_code = OneHotEncoder()
# housing_cat_code = categories_code.fit_transform(housing_cat)  # 独热编码会将文本数据改成一个0-1矩阵（一般还是稀疏矩阵）

# print(categories_code.categories_)
# print(housing_cat_code.toarray())


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


# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)
# print(housing_extra_attribs)
# -----------------------------------------------------------------------------------------


# 流水线（把前面的操作按顺序整合在一起）
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
# housing_num_tr = num_pipeline.fit_transform(housing_num)

# 自定义转换器，可以按不同的方式来同时处理不同的数据。这里用之前的流水线处理数值数据，用独热编码处理分类数据
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)  # 对指定的列进行转换（其实除了要预测的值（房价）其他都转换）
# 可以看看里面的数据元素变成啥样了(已经变成特征矩阵了)
print(housing_prepared[:5])


# -----------------------------------------选择模型并训练数据-------------------------------------------
# 首先，选择一些训练集里的数据（去标签），以便之后看预测的结果是否与标签一致
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

# # 线性回归模型
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)  # 此时，模型的斜率和截距已被计算出来了
# # print("Predictions:", lin_reg.predict(some_data_prepared))
# # print("Labels:", list(some_labels))
# # print("斜率：", lin_reg.coef_, " 截距：", lin_reg.intercept_)  # 看看参数
# # 显然，还是有一定的差错，我们来算一算RMSE
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# # print(np.sqrt(lin_mse))
# # 欠拟合了，线性模型不得用，试试看更强大的模型吧
#
#
# # 决策树回归模型
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# # print("Tree_Predictions:", tree_reg.predict(some_data_prepared))
# # print("Labels:", list(some_labels))
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# # print(np.sqrt(tree_mse))
#
#
# # 随机森林回归模型
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# # print(np.sqrt(forest_mse))
#
#
# # 我们需要使用交叉验证技术
# lin_rmse_scores = np.sqrt(-cross_val_score(lin_reg,
#                                            housing_prepared,
#                                            housing_labels,
#                                            scoring="neg_mean_squared_error",
#                                            cv=10))
# tree_rmse_scores = np.sqrt(-cross_val_score(tree_reg,
#                                             housing_prepared,
#                                             housing_labels,
#                                             scoring="neg_mean_squared_error",
#                                             cv=10))
# forest_rmse_scores = np.sqrt(-cross_val_score(forest_reg,
#                                               housing_prepared,
#                                               housing_labels,
#                                               scoring="neg_mean_squared_error",
#                                               cv=10))
#
#
# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())
#
#
# display_scores(lin_rmse_scores)
# display_scores(tree_rmse_scores)
# display_scores(forest_rmse_scores)
# 很显然，第一个欠拟合，后面两个过拟合，不过相对来说随机森林回归模型表现要好一点，暂时决定用这个模型了！


# -----------------------------------微调参数---------------------------------------
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
# print("最好的参数（不唯一）:", grid_search.best_params_)
# cvres = grid_search.cv_results_
# print("所有的参数组合:")
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)
# feature_importances = grid_search.best_estimator_.feature_importances_
# extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder = full_pipeline.named_transformers_["cat"]
# cat_one_hot_attribs = list(cat_encoder.categories_[0])
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# print("各属性的重要性:", sorted(zip(feature_importances, attributes), reverse=True))


# 最后，将找到的最好参数应用到模型中
final_model = grid_search.best_estimator_
test_no_label = strat_test_set.drop("median_house_value", axis=1)
test_label = strat_test_set["median_house_value"].copy()
test_prepared = full_pipeline.transform(test_no_label)  # 别用fit或者fit_tramsform，你也不希望纯洁的测试集被NTR吧
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
