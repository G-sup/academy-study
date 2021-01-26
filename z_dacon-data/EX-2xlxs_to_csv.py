import pandas as pd

xlsx = pd.read_excel("./z_dacon-data/sample_submission_2.xlsx",index_col=0, header=0)
xlsx.to_csv("./z_dacon-data/sample_submission_2.csv")