# importance = pd.DataFrame({'feature': list(model.get_fscore().keys()), \
#                            'importance': list(model.get_fscore().values())})\
#                             .sort_values('importance', ascending=False)
#
# importance['importance'] = importance['importance'] / importance['importance'].sum()
# columns = train_module.final_columns
# importance['feature'] = importance['feature'].map(lambda x: columns[int(x.strip('f'))])
# print(importance.head(35))
