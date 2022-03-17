from calendar import c
from cmath import exp
# import imp
import numpy as np
import pandas as pd
from math import e, pi, sqrt
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from .plots import *
import random

def cond_prob(x, mean, std):
    # gaussian distribution
    return (1/sqrt(2*pi*std**2))*np.exp(-((x - mean)**2/(2*std**2)))


class CrossValidation:

    def __init__(self, data, k):
        self._data = data
        self.k = k
        self._sets = []

    def generate_sets(self):
        self._sets.clear()
        length = len(self._data)
        set_length = length//self.k
        remaining = len(self._data) % self.k

        for i in range(self.k):
            lower = i*(set_length+1) if i < remaining else remaining*(set_length+1) + (i - remaining)*set_length
            upper = lower + set_length + (1 if i < remaining else 0)
            self._sets.append(self._data.iloc[lower:upper])
        return self._sets


class NaiveBayes:

    def __init__(self, columns, filename, classes):
        self._columns = columns
        self._classes = classes
        self._data = self._read_data(filename)


    def _read_data(self, filename):
        with open(filename) as f:
            content = f.readlines()

        all_data = [(row.split()[:-1], row.split()[-1]) for row in content]
        param, classes = zip(*all_data)
        data = np.array(param, dtype=float)
        classes = [self._classes[x] for x in classes]

        df = pd.DataFrame(data, columns = self._columns[:-1])
        df[self._columns[-1]] = classes
        return df

    def get_data(self):
        return self._data

    def count_mean_and_std(self, data):
        # count mean and std values for each attribute and class
        param = defaultdict(lambda: defaultdict(lambda:[]))
        combined = {}

        for column in data.columns[:-1]:
            combined[column] = (np.mean(data[column]), np.std(data[column]))

        # check if zero frequency problem occurs in given data
        zero_frequency = any(len(data[data[self._columns[-1]] == _class]) == 0 for _class in self._classes)

        for _class in self._classes.values():
            rows = data[data[self._columns[-1]] == _class]
            param[_class][_class] = (len(rows)+(1 if zero_frequency else 0))/(len(data)+(len(self._classes) if zero_frequency else 0))
            for column in data.columns[:-1]:
                param[_class][column] = (np.mean(rows[column]), np.std(rows[column])) if len(rows) > 0 else combined[column]
        
        return param, combined

    def classify(self, row, param, combined):
        results = {}
        for _class in self._classes.values():
            result = param[_class][_class]  # probability of given class
            for column in self._columns[:-1]:
                # multiply by (condiditonal probability of given value)/(probability of attribute)
                result *= cond_prob(row[column], *param[_class][column])/cond_prob(row[column], *combined[column])  
            results[_class] = result
        return max(results, key = results.get)

    def train_and_test(self, training_data, test_data):
        param, combined = self.count_mean_and_std(training_data)
        predicted, expected = tuple(zip(*[(self.classify(row[1], param, combined), row[1][self._columns[-1]]) for row in test_data.iterrows()]))
        return predicted, expected


def main():
    classifier = NaiveBayes(["area", "perimeter", "compactness", "length of kernel", "widht of kernel", "assymmetry coefficient", "length of kernel groove", "seed"],
        "data.txt", {"1": "Kama", "2": "Rosa", "3": "Canadian"})

    results = defaultdict(lambda: {})
    labels = sorted(list(classifier._classes.values()))
    results["classes"] = labels 

    for shuffle in (True, False):
        temp_results = defaultdict(lambda: {})
        for k in range(2, 13):
            data = deepcopy(classifier.get_data())
            if shuffle:
                data = data.sample(frac=1)
            cv = CrossValidation(data, k)
            sets = cv.generate_sets()

            print("k =", k)
            all_predicted, all_expected = [], []
            for i in range(k):
                test_data = sets[i]
                training_data = pd.concat(s for j, s in enumerate(sets) if j != i)
                predicted, expected = classifier.train_and_test(training_data, test_data)
                all_predicted.extend(predicted)
                all_expected.extend(expected)
            
            temp_results[k]["matrix"] = confusion_matrix(all_expected, all_predicted).tolist()
            temp_results[k]["accuracy"] = accuracy_score(all_expected, all_predicted)
            temp_results[k]["recall"] = {}
            temp_results[k]["precision"] = {}
            recall = recall_score(all_expected, all_predicted, average=None, labels=labels)
            precision = precision_score(all_expected, all_predicted, average=None, labels=labels)
            for i, _class in enumerate(labels):
                temp_results[k]["recall"][_class] = recall[i]       
                temp_results[k]["precision"][_class] = precision[i]  
        results["sorted" if not shuffle else "random"] = temp_results
    
    with open("results_naive_bayes.json", "w+") as f:
        json.dump(results, f)
        

if __name__ == "__main__":
    main()