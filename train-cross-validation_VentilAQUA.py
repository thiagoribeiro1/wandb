import wandb
import os
import multiprocessing
import collections
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from wandb.keras import WandbCallback
import tensorflow


dataset = pd.read_excel(r"C:\Users\thiag\Desktop\Dados gerados Bateria 2\Dados gerados Bateria r2.xlsx", "Trial 2", index_col=None, na_values=["NA"], engine="openpyxl")
dataset = dataset.iloc[:,0:11]
dataset['STATUS_bin'] = dataset['STATUS'].apply(lambda x: 1 if x == "OK" else 0)
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:,11].values


kf = KFold(n_splits=5, random_state=1, shuffle=True)

fold_no = 0
for train, test in kf.split(X, y):
    if fold_no==0:
        X_train_0 = X[train]
        X_test_0 = X[test]
        y_train_0 = y[train]
        y_test_0 = y[test]
        sc = StandardScaler()
        X_train_0 = sc.fit_transform(X_train_0)
        X_test_0 = sc.transform(X_test_0)
    elif fold_no==1:
        X_train_1 = X[train]
        X_test_1 = X[test]
        y_train_1 = y[train]
        y_test_1 = y[test]
        sc = StandardScaler()
        X_train_1 = sc.fit_transform(X_train_1)
        X_test_1 = sc.transform(X_test_1)
    elif fold_no==2:
        X_train_2 = X[train]
        X_test_2 = X[test]
        y_train_2 = y[train]
        y_test_2 = y[test]
        sc = StandardScaler()
        X_train_2 = sc.fit_transform(X_train_2)
        X_test_2 = sc.transform(X_test_2)
    elif fold_no==3:
        X_train_3 = X[train]
        X_test_3 = X[test]
        y_train_3 = y[train]
        y_test_3 = y[test]
        sc = StandardScaler()
        X_train_3 = sc.fit_transform(X_train_3)
        X_test_3 = sc.transform(X_test_3)
    else:
        X_train_4 = X[train]
        X_test_4 = X[test]
        y_train_4 = y[train]
        y_test_4 = y[test]
        fold_no = fold_no + 1
        sc = StandardScaler()
        X_train_4 = sc.fit_transform(X_train_4)
        X_test_4 = sc.transform(X_test_4)
    fold_no = fold_no + 1

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("val_accuracy"))


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

def train(sweep_q, worker_q, count):
    config_defaults = {
        "n_layers": 1,
        "layer_1": 1,
        "activation_1": "linear",
        "layer_2": 1,
        "activation_2": "linear",
        "layer_3": 1,
        "activation_3": "linear",
        "layer_4": 1,
        "activation_4": "linear",
        "layer_5": 1,
        "activation_5": "linear",
        "optimizer": "adam",
        "epoch": 100        
    }

    reset_wandb_env()
    worker_data = worker_q.get()
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.num)
    # config = worker_data.config
    run = wandb.init(
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config_defaults,
        settings=wandb.Settings(start_method="thread")
    )
    
    config = worker_data.config
    
    model = Sequential()

    if config['n_layers']==1:
        model.add(Dense(units=config['layer_1'], activation=config['activation_1']))    
    if config['n_layers']==2:
        model.add(Dense(units=config['layer_1'], activation=config['activation_1']))
        model.add(Dense(units=config['layer_2'], activation=config['activation_2']))
    if config['n_layers']==3:
        model.add(Dense(units=config['layer_1'], activation=config['activation_1']))
        model.add(Dense(units=config['layer_2'], activation=config['activation_2']))
        model.add(Dense(units=config['layer_3'], activation=config['activation_3']))
    if config['n_layers']==4:
        model.add(Dense(units=config['layer_1'], activation=config['activation_1']))
        model.add(Dense(units=config['layer_2'], activation=config['activation_2']))
        model.add(Dense(units=config['layer_3'], activation=config['activation_3']))
        model.add(Dense(units=config['layer_4'], activation=config['activation_4']))
    if config['n_layers']==5:
        model.add(Dense(units=config['layer_1'], activation=config['activation_1']))
        model.add(Dense(units=config['layer_2'], activation=config['activation_2']))
        model.add(Dense(units=config['layer_3'], activation=config['activation_3']))
        model.add(Dense(units=config['layer_4'], activation=config['activation_4']))
        model.add(Dense(units=config['layer_5'], activation=config['activation_5']))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer=config['optimizer'], metrics=['accuracy', 'AUC'])
    
    if count==0:
        history = model.fit(X_train_0, y_train_0, batch_size=64, epochs=config['epoch'], validation_data=(X_test_0, y_test_0), callbacks=[WandbCallback()])
    elif count==1:
        history = model.fit(X_train_1, y_train_1, batch_size=64, epochs=config['epoch'], validation_data=(X_test_1, y_test_1), callbacks=[WandbCallback()])
    elif count==2:
        history = model.fit(X_train_2, y_train_2, batch_size=64, epochs=config['epoch'], validation_data=(X_test_2, y_test_2), callbacks=[WandbCallback()])
    elif count==3:
        history = model.fit(X_train_3, y_train_3, batch_size=64, epochs=config['epoch'], validation_data=(X_test_3, y_test_3), callbacks=[WandbCallback()])
    else:
        history = model.fit(X_train_4, y_train_4, batch_size=64, epochs=config['epoch'], validation_data=(X_test_4, y_test_4), callbacks=[WandbCallback()])
    
    run.log(dict(val_accuracy=history.history['val_accuracy']))
    # wandb.join()
    wandb.finish()
    sweep_q.put(WorkerDoneData(val_accuracy=history.history['val_accuracy']))


def main():
    count = 0
    num_folds = 5

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    for num in range(num_folds):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train, kwargs=dict(sweep_q=sweep_q, worker_q=q, count=count)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))
        count = count + 1

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    metrics = []
    for num in range(num_folds):
        worker = workers[num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics.append(result.val_accuracy)

    sweep_run.log(dict(val_accuracy=sum(metrics) / len(metrics)))
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


if __name__ == "__main__":
    main()