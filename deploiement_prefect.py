"""Déploiement Prefect : lancement planifié des flows (piste d'excellence, étape 05).

Terminal 1 : prefect server start
Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
             python deploiement_prefect.py
Terminal 3 : prefect deployment ls
             prefect deployment run 'all/ml-pipeline-all'
"""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import flow_all, flow_evaluate, flow_train

if __name__ == "__main__":
    serve(
        flow_all.to_deployment(
            name="ml-pipeline-all",
            schedules=[CronSchedule(cron="0 2 * * *", timezone="Africa/Tunis")],
            tags=["mlops", "full-pipeline"],
        ),
        flow_train.to_deployment(
            name="ml-pipeline-train",
            schedules=[CronSchedule(cron="0 3 * * *", timezone="Africa/Tunis")],
            tags=["mlops", "train"],
        ),
        flow_evaluate.to_deployment(name="ml-pipeline-evaluate", tags=["mlops", "evaluate"]),
    )
