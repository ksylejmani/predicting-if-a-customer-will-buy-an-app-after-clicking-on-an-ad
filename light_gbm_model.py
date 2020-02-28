import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# *** Importing data
file_path = 'input/train_sample.csv'
clicks = pd.read_csv(file_path, parse_dates=['click_time'])

# Prepping date-time variables
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')

# Prepping categorical variables
cat_features = ['ip', 'app', 'device', 'os', 'channel']
label_encoder = LabelEncoder()
for cat in cat_features:
    clicks[cat + '_labels'] = label_encoder.fit_transform(clicks[cat])

# Split the data set
valid_fraction = 0.1
clicks_srt = clicks.sort_values('click_time')
valid_rows = int(len(clicks_srt) * valid_fraction)
train = clicks_srt[0:len(clicks_srt) - valid_rows * 2]
valid = clicks_srt[len(clicks_srt) - valid_rows * 2:len(clicks_srt) - valid_rows]
test = clicks_srt[len(clicks_srt) - valid_rows:len(clicks_srt)]
feature_cols = ['day', 'hour', 'minute', 'second',
                'ip_labels', 'app_labels', 'device_labels',
                'os_labels', 'channel_labels']
X_train = train[feature_cols]
y_train = train['is_attributed']
X_valid = valid[feature_cols]
y_valid = valid['is_attributed']
X_test = test[feature_cols]
y_test = test['is_attributed']

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid)
dtest = lgb.Dataset(X_test, label=y_test)

# Train the model
param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)

# Predict using the model
ypred = bst.predict(test[feature_cols])
score = roc_auc_score(y_test, ypred)
print(f"Test score: {score}")
