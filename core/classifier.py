import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os


class Classifier:
    def __init__(self, n_estimators: int = 50, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.features = ["mean_r", "mean_g", "mean_b", "std_dev", "gradient"]
        self.trained = False   

### Treinar Modelo
    def train(self, df: pd.DataFrame):
        df_train = df[(df["label"].notna()) & (df["label"] != "")].copy()
        
        if len(df_train) == 0:
            raise ValueError("Não há dados de treinamento disponíveis.")

        x_train = df_train[self.features]
        y_train = df_train["label"]

        self.model.fit(x_train, y_train)
        self.trained = True
        
    
### Salvar modelo treinado em um arquivo pickle
    def save_model(self, model_path: str = "./data/model/random_forest_trainned.pkl"):
        if not self.trained:
            raise ValueError("O modelo não foi treinado. Treine o modelo antes de salvá-lo.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"[X] Modelo salvo com sucesso em {str}")
       
#Carregar Modelo treinado a partir de um arquivo pickle
    def load_model(self, model_path: str = "./data/model/random_forest_trainned.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado no caminho especificado: {model_path}")
        self.model = joblib.load(model_path)
        self.trained = True
        print(f"Modelo carregado com sucesso de: {model_path}")
        
        
 ### Predizer/Classificar os dados através do treinamento do modelo
    def predict(self, df: pd):
        if not self.trained:
            raise ValueError("O modelo não foi treinado. Treine o modelo antes de fazer previsões.")
        return self.model.predict(df[self.features])