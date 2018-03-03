import os
import sys
import xgbfir

import pandas as pd
import xgboost as xgb

sys.path.append("../src")

from base import Utilities, Config
from common import CustomTransformation

config = Config()

train_module = CustomTransformation(config, 'train')
watchlist = [(train_module.ddata, 'train')]

print(train_module.final_columns)

params = Utilities.load_json(config.params_file)
history = xgb.cv(params, train_module.ddata, 300, early_stopping_rounds=30, metrics=["auc", "error"], verbose_eval=True)

model = xgb.train(params, train_module.ddata, 200, verbose_eval=True)

class_mapping = Utilities.load_json(config.class_mapping_file)
test_module = CustomTransformation("test", class_mapping, train_module.final_columns)
y_pred = model.predict(test_module.ddata)
submission_df = pd.DataFrame({config.notable_columns["ID"]: list(test_module.main_column.values),
                              config.notable_columns["Target"]: list(y_pred)})
submission_df.to_csv(os.path.join(config.home, 'submission', 'one.csv'), float_format='%0.6f', index=False)

xgbfir.saveXgbFI(model, feature_names=train_module.final_columns, TopK=500, SortBy='Gain', \
                 MaxTrees=500, MaxInteractionDepth=2, OutputXlsxFile='XGBoost-FI.xlsx')
