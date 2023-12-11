from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from loanPrediction import logger
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from box import ConfigBox
from sklearn.svm import SVC
from loanPrediction.utils.common import save_bin, create_directories
from loanPrediction.entity.config_entity import ModelTrainerConfig
from pathlib import Path
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        train = pd.read_csv(self.config.train_data_path)
        test = pd.read_csv(self.config.test_data_path)

        self.X_train = train.drop(["Loan_Status"], axis=1)
        self.y_train = train["Loan_Status"]
        self.X_test = test.drop(["Loan_Status"], axis=1)
        self.y_test = test["Loan_Status"]

    def _randomized_search(self, name, clf, params, runs=20):
        rand_clf = RandomizedSearchCV(
            clf, params, n_iter=runs, cv=5, n_jobs=-1, random_state=2
        )

        rand_clf.fit(self.X_train, self.y_train)
        best_model = rand_clf.best_estimator_

        # Extract best score
        best_score = rand_clf.best_score_

        # Print best score
        logger.info("Trained with {} with score: {:.3f}".format(name, best_score))

        # Predict test set labels
        y_pred = best_model.predict(self.X_test)

        # Compute accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        # Print accuracy
        logger.info("Predicted with {} ; Test score : {:.3f}".format(name, accuracy))

        return best_model, accuracy

    def train(self):
        model_params = self.config.params

        models = ConfigBox(
            {
                "Decision_Tree": {
                    "model": DecisionTreeClassifier(),
                    "params": model_params.Decision_Tree,
                },
                "Random_Forest": {
                    "model": RandomForestClassifier(),
                    "params": model_params.Random_Forest,
                },
                "SVC": {"model": SVC(), "params": model_params.SVC},
            }
        )

        create_directories([os.path.join(self.config.root_dir, "models")])
        trained_models = []
        for model in models:
            clf = models[model].model
            params = models[model].params

            clf_model, score = self._randomized_search(
                name=str(model), clf=clf, params=params
            )
            trained_models.append((clf_model, score))

            save_bin(
                data=clf_model,
                path=Path(
                    os.path.join(self.config.root_dir, f"models/{str(model)}.joblib")
                ),
            )

        trained_models = sorted(
            trained_models, key=lambda x: x[1], reverse=True
        )  # [(model, score), (model, score), ..]
        best_model = trained_models[0][0]  # taking the model

        save_bin(
            data=best_model,
            path=Path(os.path.join(self.config.root_dir, self.config.model_name)),
        )

        best_model_name = str(best_model)[: str(best_model).find("(")]
        best_model_score = round(trained_models[0][1], 3)
        logger.info(
            f"Saved main model as {best_model_name}, with score - {best_model_score}"
        )
