# Градиентный бустинг для бинарной классификации

## Простое использование:

```python
from boosting import *

boosting = GradBoostingBinaryClassifier()
boosting.fit(X_train, y_train)
preds = np.round(boosting.predict(X_test) > 0.5)
```

## Подбор гиперпараметров:
```python
from boosting import *
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

class HPOpt(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials
    
    def boosting(self, para):
        clf = GradBoostingBinaryClassifier(**para['model_params'])
        return self.train_boosting(clf, para)
    
    def train_boosting(self, clf, para):
        clf.fit(self.x_train, self.y_train)
        pred = clf.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}

sigmoid = lambda z: 1 / (1 + np.exp(-z))

boosting_model_params = {
    'subsample':        hp.uniform('subsample', 0.1, 1),
    'n_estimators':     hp.choice('n_estimators', np.arange(1, 100, 20)),
    'learning_rate':    hp.uniform('learning_rate', 0.1, 1),
}

boosting_params = dict()
boosting_params['model_params'] = boosting_model_params
sigmoid = lambda z: 1 / (1 + np.exp(-z))
boosting_params['loss_func' ] = lambda y, z: -np.log(sigmoid(y * z)).mean()

obj = HPOpt(X_train, X_val, y_train, y_val)
boosting_opt = obj.process(fn_name='boosting', space=boosting_params, trials=Trials(), algo=tpe.suggest, max_evals=100)
```
