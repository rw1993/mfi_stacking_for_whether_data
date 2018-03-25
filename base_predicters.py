from get_data import data, cities
from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor)
BasePredicters = [("Ada", AdaBoostRegressor),
                  ("GBDT", GradientBoostingRegressor),
                  ("RF", RandomForestRegressor)]
import copy
import numpy as np
import pickle


def generate_XY_from_data(data_for="test", last_days=10):
    total_days = data[0].shape[0]
    train_valid = int(total_days*0.5)
    valid_test = int(total_days*0.8)

    def generate_XY_from_subdata(subdata):
        X = []
        Y = []
        if data_for == "train":
            d = subdata[:train_valid]
        elif data_for == "valid":
            d = subdata[train_valid: valid_test]
        else:
            d = subdata[valid_test: ]
        
        for index in range(d.shape[0]):
            if index - last_days < 0:
                continue
            Y.append(d[index][0])
            x = copy.copy(d[index-last_days: index, :]).reshape(-1)
            X.append(x)
        return np.array(X), np.array(Y)
    XYS = list(map(generate_XY_from_subdata, data))
    Xs = [X  for X, Y in XYS]
    Ys = [Y for X, Y in XYS]
    return np.vstack(Xs), np.concatenate(Ys), Xs[0].shape

    
def main():
    X, Y, _ = generate_XY_from_data(data_for="train")
    for name, Predicter in BasePredicters:
        print(Predicter)
        p = Predicter()
        p.fit(X, Y)
        print("finished")
        with open("BasePredicters/{}_predicter".format(name), "wb") as f:
            pickle.dump(p, f)


if __name__ == '__main__':
    main()
