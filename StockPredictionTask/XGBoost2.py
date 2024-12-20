import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_path = 'F:\\机器学习大作业\\Personal\\data\\train.csv'
test_path = 'F:\\机器学习大作业\\Personal\\data\\test.csv'
# 读取CSV文件
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
label_encoder = LabelEncoder()
train_df['Sector'] = label_encoder.fit_transform(train_df['Sector'])
test_df['Sector'] = label_encoder.fit_transform(test_df['Sector'])
features1 = [col for col in train_df.columns if col not in ['id', 'Class']]
features = ['Weighted Average Shares Diluted Growth', 'Net cash flow / Change in cash', 'Book Value per Share Growth',
            'SG&A Expenses Growth', 'Operating Cash Flow growth', '3Y Revenue Growth (per Share)',
            'Receivables growth',
            'Revenue Growth',
            'Free Cash Flow growth',
            'Weighted Average Shares Growth',
            '5Y Revenue Growth (per Share)',
            'Asset Growth',
            'Weighted Average Shs Out',
            '3Y Shareholders Equity Growth (per Share)',
            'Retained earnings (deficit)',
            '5Y Shareholders Equity Growth (per Share)',
            'Gross Profit Growth',
            'Issuance (buybacks) of shares',
            'Debt Growth',
            'Other comprehensive income',
            'Operating Income Growth',
            'Net Cash/Marketcap',
            'Other Assets',
            'Financing Cash Flow',
            'cashFlowToDebtRatio',
            'Capex to Depreciation',
            'niperEBT',
            'cashRatio'
            ]
target = ['Class']
# 划分测试集
X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df[target], test_size=0.2,
                                                    random_state=42)
# 预测数据准备
test_ids = test_df['id'].values
prediction_data = xgb.DMatrix(test_df[features])
dtrain = xgb.DMatrix(X_train, label=y_train)
# 设置 XGBoost 的参数
params = {
    'objective': 'binary:logistic',  # 二分类任务
    'eval_metric': 'error',  # 评估指标使用 logloss
    'max_depth': 10,  # 树的最大深度
    'eta': 0.1,  # 学习率
    'silent': 1,  # 训练时不输出
    'tree_method': 'gpu_hist',  # 使用 GPU 加速的历史方法
    'gpu_id': 0,  # 使用 GPU 设备编号，通常 GPU 编号从 0 开始
    'max_bin': 256,  # GPU 加速时的 bin 数量，越大越精确但越慢
    'min_child_weight': 3,
    'n_estimators': 8,
    'gamma': 0.1,
    'lambda': 1,  # L2 正则化项
    'alpha': 0.5,  # L1 正则化项
}

num_round = 20
bst = xgb.train(params, dtrain, num_round)
# 测试
test_result = bst.predict(xgb.DMatrix(X_test))
test_result = (test_result > 0.5).astype(int)
TEST_result = bst.predict(xgb.DMatrix(X_train))
TEST_result = (TEST_result > 0.5).astype(int)
print("AccuracyTest:", accuracy_score(y_train, TEST_result))
print("Accuracy:", accuracy_score(y_test, test_result))

# 预测
preds = bst.predict(prediction_data)
predictions = (preds > 0.5).astype(int)
submission_df = pd.DataFrame({'id': test_ids, 'Class': predictions})
submission_df.to_csv('F:\\机器学习大作业\\Personal\\output\\submission.csv', index=False)
