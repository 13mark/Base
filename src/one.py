import xgboost as xgb

from base import Config, Utilities
from common import CustomTransformation

config = Config()
train_module = CustomTransformation(config, 'train')

# watchlist = [(train_module.ddata, 'train')]

params = Utilities.load_json(config.params_file)

print(len(train_module.final_columns))

model = xgb.cv(params, train_module.ddata, 300, early_stopping_rounds=30, metrics=["auc", "error"], verbose_eval=True)
#model = xgb.cv(params, train_module.ddata, 100, early_stopping_rounds=10, verbose_eval=True)