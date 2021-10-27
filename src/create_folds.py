from config import config
import pandas as pd
from sklearn import model_selection

def make_folds(dataset_path):
    dfx = pd.read_csv(dataset_path).dropna().reset_index(drop = True)

    dfx["target"] = dfx["language"].apply(lambda x : 1 if x == "tamil" else 0)

    dfx["kfold"] = -1    
    dfx = dfx.sample(frac=1).reset_index(drop=True)
    y = dfx.target.values
    kf = model_selection.KFold(n_splits=5) # KFold for regression problems

    for f, (t_, v_) in enumerate(kf.split(X=dfx, y=y)):
        dfx.loc[v_, 'kfold'] = f

    return dfx

if __name__ == "__main__":
    training_data_path = config.TRAINING_FILE

    df = make_folds(training_data_path)
    print(len(df[df['kfold'] == 4]))
