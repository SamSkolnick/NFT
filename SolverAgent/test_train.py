from train import train_model

data_path = 'data/spam/train.csv'
valdata_path = 'data/spam/test.csv'
target_col = 'target'

results = train_model(
    task_desc="This is a classificaiton task where you determine if the emails are spam or ham.",
    constraints="Make it simple and fast. Maximize validation accuracy.",
    data_path=data_path,
    valdata_path=valdata_path,
    target_col=0,
    )

r = results 