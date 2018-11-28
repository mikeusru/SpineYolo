from spine_preprocessing.collect_spine_data import SpineImageDataPreparer

app = SpineImageDataPreparer()
app.create_dataframe()
app.run()

