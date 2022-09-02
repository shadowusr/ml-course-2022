from config import config
from auto_ml import AutoML
from prettytable import PrettyTable


reports = AutoML.get_reports(config)

table = PrettyTable()

table.field_names = ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "Time (s)"]
for index, report in enumerate(reports):
    table.add_row([
        str(config["models"][index]),
        report["accuracy"],
        report["precision"],
        report["recall"],
        report["specificity"],
        report["f1_score"],
        report["time"]
    ])

print(table)
