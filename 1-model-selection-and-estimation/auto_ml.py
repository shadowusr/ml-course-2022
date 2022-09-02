import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, ShuffleSplit
from timeit import default_timer as timer

class AutoML:
    @staticmethod
    def __get_metrics_by_class(confusion_matrix: np.array):
        # A good article on confusion matrix:
        # https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
        metrics = []
        classes_count = confusion_matrix.shape[0]

        for i in range(classes_count):
            tp = confusion_matrix[i][i]
            tn = np.sum(
                np.delete(
                    np.delete(confusion_matrix, i, 0),
                    i, 1
                )
            )
            fp = np.sum(np.delete(confusion_matrix[:, i], i, 0))
            fn = np.sum(np.delete(confusion_matrix[i], i, 0))

            metrics.append({
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "f1_score": (2 * tp) / (2 * tp + fp + fn),
            })

        return metrics

    @staticmethod
    def __get_average_metrics(metrics_by_class):
        result = {}
        metrics = ["accuracy", "precision", "recall", "specificity", "f1_score"]

        for metric in metrics:
            result[metric] = sum(item[metric] for item in metrics_by_class) / len(metrics_by_class)

        return result

    @staticmethod
    def get_reports(config):
        reports = []
        X = config["dataset"]["samples"]
        y = config["dataset"]["labels"]
        y_count = len(set(y))

        for model in config["models"]:
            start_time = timer()
            print(f"Evaluating model {model}")

            labels_dict = {k: v for v, k in enumerate(set(y))}
            label_to_index = lambda labels: [labels_dict.get(label) for label in labels]

            confusion_matrix = np.zeros((y_count, y_count))

            split = None

            if config["validation"]["method"] == "loo":
                loo = LeaveOneOut()
                split = loo.split(X)
            elif config["validation"]["method"] == "k_fold":
                kf = KFold(n_splits=config["validation"]["n_splits"])  # 2
                split = kf.split(X)
            elif config["validation"]["method"] == "shuffle_split":
                rs = ShuffleSplit(
                    n_splits=config["validation"]["n_splits"],  # 5
                    test_size=config["validation"]["test_size"],  # 0.25
                    random_state=0
                )
                split = rs.split(X)
                pass

            iteration = 0
            for train_index, test_index in split:
                print(f"\riteration: {iteration}", end='')
                X_train, X_test = X.loc[train_index], X.loc[test_index]
                y_train, y_test = y.loc[train_index], y.loc[test_index]

                # Calling fit() multiple times on the same model is fine:
                # https://stackoverflow.com/questions/49841324/what-does-calling-fit-multiple-times-on-the-same-model-do
                model.fit(X_train, y_train)

                y_predicted = model.predict(X_test)
                confusion_matrix[label_to_index(y_test), label_to_index(y_predicted)] += 1

                iteration += 1

            print("\n")
            end_time = timer()

            report = AutoML.__get_average_metrics(AutoML.__get_metrics_by_class(confusion_matrix))
            report["time"] = end_time - start_time

            reports.append(report)

        return reports
