from scipy.stats import truncnorm
import numpy as np
MAX_TEMP = 50
MIN_TEMP = -30

def classification_to_regression(probas):
    step = (MAX_TEMP - MIN_TEMP) / len(probas)
    index = np.array(probas).argmax()
    return step * index + step / 2 + MIN_TEMP


def get_feature_slot(step=5):
    feature = []
    index = MIN_TEMP
    while index < MAX_TEMP:
        feature.append(0.0)
        index += step
    return feature

def simple_regression_to_classsification(regression_result, step=5):
    tmp = regression_result - MIN_TEMP
    index = tmp / step
    feature = get_feature_slot(step)
    feature[int(index)] = 1.0
    return feature

def truncated_normal_regression_to_classification(regression_result,
                                                  scale=1, step=5):
    a = truncnorm(a=MIN_TEMP, b=MAX_TEMP, loc=regression_result,
                  scale=scale)
    feature = []
    former_cdf = 0.0
    begin = MIN_TEMP
    while begin < MAX_TEMP:
        begin += step
        this_cdf = a.cdf(begin)
        feature.append(this_cdf-former_cdf)
        former_cdf = this_cdf
    return feature
    
def main():
    #feature = find_feature_slot(2.0, step=5)
    #feature = truncated_normal_regression_to_classification(regression_result=1.0)
    classification_to_regression([0.1, 0.5, 0.3, 0.1])

    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()