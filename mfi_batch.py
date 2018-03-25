import pickle
from base_predicters import BasePredicters, generate_XY_from_data
from get_data import cities
from regression_to_classification import truncated_normal_regression_to_classification as rtc
import numpy as np

Regressors = [pickle.load(open("BasePredicters/{}_predicter".format(name),
                         "rb")) for name, P in BasePredicters]

def generate_data(data_for="valid"):
    X, Y, _ = generate_XY_from_data(data_for="valid")
    results = [regressor.predict(X) for regressor in Regressors]
    avg_results = sum(results) / len(results)
    days = int(Y.shape[0] / len(cities))
    total = []
    for d in range(days):
        print(d)
        indexs = [d+i*days for i in range(len(cities))]
        y = Y[indexs]
        avg_result = avg_results[indexs]
        q = [rtc(avg_result[i], step=2, scale=1) for i in range(avg_result.shape[0])]
        q = np.array(q)
        f = np.array([r[indexs] for r in results]).T
        total.append((y, q, f, avg_result))
    with open("data/mfi_{}_batch".format(data_for), "wb") as f:
        pickle.dump(total, f)

def main():
    generate_data("valid")
    generate_data("test")

if __name__ == '__main__':
    main()

