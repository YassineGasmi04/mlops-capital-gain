import datetime
import numpy as np
import pandas as pd
import sklearn
import mlflow

print("Étudiant : Yassine Gasmi")
print("Date     :", datetime.datetime.now().strftime("%d/%m/%Y %H:%M"))
print("numpy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__, "| mlflow", mlflow.__version__)
print("Environment is set up correctly!")
