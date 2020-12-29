from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np


class GradBoostingBinaryClassifier:
    """
    Градиентный бустинг для бинарной классификации
    """
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: dict={'max_features': 0.1},
        n_estimators: int=10,
        learning_rate: float=0.1,
        subsample: float=0.3,
        random_seed: int=228,
        custom_loss: list or tuple=None,
        use_best_model: bool=True,
        n_iter_early_stopping: int=None
    ):
        
        # Класс базовой модели
        self.base_model_class = base_model_class
        # Параметры для инициализации базовой модели
        self.base_model_params = base_model_params
        # Число базовых моделей
        self.n_estimators = n_estimators
        # Длина шага (которая в лекциях обозначалась через eta)
        self.learning_rate = learning_rate
        # Доля объектов, на которых обучается каждая базовая модель
        self.subsample = subsample
        # seed для бутстрапа, если хотим воспроизводимость модели
        self.random_seed = random_seed
        # Использовать ли при вызове predict и predict_proba лучшее
        # с точки зрения валидационной выборки число деревьев в композиции
        self.use_best_model = use_best_model
        # число итераций, после которых при отсутствии улучшений на валидационной выборке обучение завершается
        self.n_iter_early_stopping = n_iter_early_stopping
        
        # Плейсхолдер для нулевой модели
        self.initial_model_pred = None
        
        # Список для хранения весов при моделях
        self.gammas = []
        
        # Создаем список базовых моделей
        self.models = [self.base_model_class(**self.base_model_params) for _ in range(self.n_estimators)]
        
        # Если используем свою функцию потерь, ее нужно передать как список из loss-a и его производной
        if custom_loss is not None:
            self.loss_fn, self.loss_derivative = custom_loss
        else:
            self.sigmoid = lambda z: 1 / (1 + np.exp(-z))
            self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
            self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        
        
    def _fit_new_model(self, X: np.ndarray, y: np.ndarray or list, n_model: int):
        """
        Функция для обучения одной базовой модели бустинга
        :param X: матрица признаков
        :param y: вектор целевой переменной
        :param n_model: номер модели, которую хотим обучить
        """

        np.random.seed(self.random_seed+n_model) # чтобы вроде как рандом, но воспроизводимость была
        indexes = np.random.choice(len(X), int(self.subsample * len(X)))
        self.models[n_model].fit(X[indexes],-self.loss_derivative(y[indexes],self.residual[indexes]))
        self.gammas.append(self._find_optimal_gamma(y[indexes],self.residual[indexes],self.models[n_model].predict(X[indexes])))
        
    def _fit_initial_model(self, X, y):
        """
        Функция для построения нулевой (простой) модели.
        :param X: матрица признаков
        :param y: вектор целевой переменной
        """
        
        uni , counts = np.unique(y, return_counts=True)
        self.uni_classes = uni
    
        #доля класса 1 (если у нас больше класса -1 то мы получим значение r от 0 до 0.5 что даст отрицательное число,
        # иначе от 0.5 до 1 что даст положительное значение)
        q = counts[1]/(counts[1]+counts[0])
        self.initial_model_pred = -np.log(1/q - 1)
    
    
    def _find_optimal_gamma(self, y: np.ndarray or list, old_predictions: np.ndarray,
                            new_predictions: np.ndarray, boundaries: tuple or list=(0.01, 1)):
        """
        Функция для поиска оптимального значения параметра gamma (коэффициент перед новой базовой моделью).
        :param y: вектор целевой переменной
        :param old_predictions: вектор суммы предсказаний предыдущих моделей (до сигмоиды)
        :param new_predictions: вектор суммы предсказаний новой модели (после сигмоиды)
        :param boudnaries: в каком диапазоне искать оптимальное значение ɣ (array-like объект из левой и правой границ)
        """
        # Определеяем начальные лосс и оптимальную гамму
        loss, optimal_gamma = self.loss_fn(y, old_predictions), 0
        # Множество, на котором будем искать оптимальное значение гаммы
        gammas = np.linspace(*boundaries, 100)
        # Простым перебором ищем оптимальное значение
        for gamma in gammas:
            predictions = old_predictions + gamma * new_predictions
            gamma_loss = self.loss_fn(y, predictions)
            if gamma_loss < loss:
                optimal_gamma = gamma
                loss = gamma_loss
        
        return optimal_gamma
        
        
    def fit(self, X, y, eval_set=None):
        """
        Функция для обучения всей модели бустинга
        :param X: матрица признаков
        :param y: вектор целевой переменной
        :eval_set: кортеж (X_val, y_val) для контроля процесса обучения или None, если контроль не используется
        """

        np.random.seed(self.random_seed)
        self._fit_initial_model(X,y)
        self.residual = np.array([self.initial_model_pred]*X.shape[0])
        count_not_improve_iter = 0
        prev_metric = 0

        for i in range(self.n_estimators):
            # сколько итераций прошло на самом деле
            self.real_n_estimators = i + 1
            self._fit_new_model(X,y,i)
            self.residual = self.residual + self.learning_rate * self.gammas[i] * self.models[i].predict(X)
            if eval_set is not None:
                pred_val = self.predict(eval_set[0])
                if accuracy_score(eval_set[1],np.round(pred_val))<=prev_metric:
                    count_not_improve_iter+=1
                else:
                    count_not_improve_iter = 0
                prev_metric = accuracy_score(eval_set[1],np.round(pred_val))
                if count_not_improve_iter==self.n_iter_early_stopping:
                    if self.use_best_model:
                        # уже давольно много if, но we need to go deeper
                        self.real_n_estimators-=count_not_improve_iter
                    print('Early stop. Optimal number of estimators = ',self.real_n_estimators)
                    break

        
    def predict(self, X: np.ndarray):
        """
        Функция для предсказания классов обученной моделью бустинга
        :param X: матрица признаков
        """
        
        results = self.initial_model_pred
        for i in range(self.real_n_estimators):
            results = results + self.learning_rate * self.gammas[i] * self.models[i].predict(X)
        return np.where(results>0, self.uni_classes[1], self.uni_classes[0])
        
    def predict_proba(self, X: np.ndarray):
        """
        Функция для предсказания вероятностей классов обученной моделью бустинга
        :param X: матрица признаков
        """

        results = self.initial_model_pred 
        for i in range(self.real_n_estimators):
            results = results + self.learning_rate * self.gammas[i] * self.models[i].predict(X)
        return np.array([1-self.sigmoid(results),self.sigmoid(results)]).T
        
    @property
    def feature_importances_(self):
        """
        Функция для вычисления важностей признаков.
        Вычисление должно проводиться после обучения модели
        и быть доступно атрибутом класса.

        !!!! Пока работает только для классов имеющих функцию feature_importances_
        """

        feat_import = np.array([model.feature_importances_ for model in self.models[:self.real_n_estimators]]).T
        feat_import = np.array([feat.mean() for feat in feat_import])
        feat_import = feat_import/np.sum(feat_import)
        return feat_import