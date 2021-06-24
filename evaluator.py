import numpy as np
from shutil import copyfile
np.random.seed(2021)


class Evaluator:
    def __init__(self):
        self.eval_scores = []

    def _evaluate(self, model, training_set, validation_set):
        model.fit(training_set)
        predictions = model.predict(validation_set[0])
        self.eval_scores.append(self._compute_score(validation_set[1], predictions))

    def evaluate(self, model):
        data_iterator = iter(model.data)
        for training_set, validation_set in data_iterator:
            self._evaluate(model, training_set, validation_set)
        self._report(model)

    def _report(self, model):
        # 把模型若干次 evaluate 的结果写到表格
        try:
            f = open('eval_scores.csv', 'x')
            f.write('ID,'+','.join(str(e) for e in range(1, len(self.eval_scores)+1))+',mean,std,std/mean'+'\n')
        except FileExistsError:
            f = open('eval_scores.csv', 'a')

        mean = np.mean(self.eval_scores)
        std = np.std(self.eval_scores)
        f.write(model.id[:4]+','+','.join(f'{e:.2f}' for e in self.eval_scores)+','+','.join([f'{mean:.2f}',f'{std:.2f}',f'{std/mean:.2f}'])+'\n')
        f.close()
        copyfile('model.py', f'model_definitions/definition:{model.id[:4]}.py')

    def _compute_score(self, true, predictions):
        return np.sqrt(np.mean((true - predictions) ** 2))
