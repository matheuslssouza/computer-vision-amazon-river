import pandas as pd
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def trainning_model(sheet_path, SLICE):
    """Treina o modelo de aprendizado de máquina."""
    # Implementação do treinamento do modelo
    
    df = pd.read_excel(sheet_path)  # Exemplo de leitura do arquivo Excel
    df_treino = df[df["tipo"].notna() & (df["tipo"] != "")].copy()
    
    treino_x = df_treino[["Mean Red", "Mean Green", "Mean Blue", "DesvPad", "Gradient"]]
    treino_y = df_treino["tipo"]
    
    ## Treinamento do modelo Random Forest
    modelo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo.fit(treino_x, treino_y)
    
    print("Modelo treinado com sucesso!")
    
    ### Carregar Imagem
    img_original = Image.open("./teste.jpg").convert("RGB")
    mapa_pintado = np.array(img_original, dtype=np.float32)
    
    # Previsão em lote: usamos a IA em TODAS as linhas da planilha de uma só vez!
    x_tudo = df[["Mean Red", "Mean Green", "Mean Blue", "DesvPad", "Gradient"]]
    predicoes = modelo.predict(x_tudo)
    
    # Pintar o mapa com base nas previsões
    COR_RIO = np.array([0, 0, 255])  # Azul
    COR_MATA = np.array([0, 255, 0])  # Verde
    
    for idx, row in df.iterrows():
        x = row["coordenada_x"]
        y = row["coordenada_y"]
        tipo_predito = predicoes[idx]
        bloco_original = mapa_pintado[y:y+SLICE, x:x+SLICE]
        
        if tipo_predito == "rio":
            mapa_pintado[y:y+SLICE, x:x+SLICE] = COR_RIO * 0.5 + 0.5 * bloco_original  # Mistura com a cor original
        elif tipo_predito == "mata":
            mapa_pintado[y:y+SLICE, x:x+SLICE] = COR_MATA * 0.5 + 0.5 * bloco_original  # Mistura com a cor original
    
    # Salvar a imagem pintada
    img_pintada = Image.fromarray(np.uint8(mapa_pintado))
    img_pintada.save("./data/images/mapa_pintado.png")
    print("Mapa pintado salvo em './data/images/mapa_pintado.png'")
    