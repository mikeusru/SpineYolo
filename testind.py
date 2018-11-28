from spine_preprocessing.collect_spine_data import SpineImageDataPreparer

app = SpineImageDataPreparer()
app.create_dataframe()
row = app.dataframe.iloc[0]