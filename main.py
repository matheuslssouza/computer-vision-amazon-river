import numpy as np
import os
from PIL import Image
import pandas as pd

from core.window    import SlidingWindow
from core.features  import FeatureExtractor
from core.classifier import Classifier
from core.renderer import MapRenderer

"""
Abrir imagem e colocar na memoria em array numpy
"""

def main():
    SLICE = 16  # Altura da fatia em pixels
    img_path = ""
    features = ["mean_r", "mean_g", "mean_b", "std_dev", "gradient"]
    
    ### Choose Image
    while True:
        img_path = input("Insira o caminho da imagem (pressione Enter para './teste.jpg')").strip()
        if img_path == "":
            img_path = "./teste.jpg"
        if os.path.exists(img_path):
            break
        print("A imagem não existe! Insira um caminho que exista!")
        
    img_pill = Image.open(img_path).convert("RGB")
    original_img = np.array(img_pill, dtype=np.float32)
    
    ### Variaveis Globais
    classifier = Classifier()
    
    ### Options Loop 
    while True:
        print("\n" + "="*35)
        print("    PAINEL DE CLASSIFICAÇÃO")
        print("="*35)
        print("1 - Extrair caracteristicas de uma imagem e salvar dados da extracao")
        print("2 - Treinar modelo")
        print("3 - Abrir (carregar) modelo existente")
        print("4 - Identificar e mapear")
        print("5 - Sair")
        print("="*35)
        
        option = input("Digite a opção desejada (1-6): ").strip()
         
        match option:
            case "1":
                ### Extrair caracteristicas e fazer planilha!
                window = SlidingWindow()
                rows = []

                for patch, patch_id, x, y in window.slice_image(original_img):
                    metrics = FeatureExtractor.extract_features(patch, patch_id, x, y)
                    rows.append(metrics.to_dict())
                
                ### Criar Df excel
                archive_name = input("Digite o nome do arquivo excel para salvar as caracteristicas extraidas: ").strip()
                os.makedirs("./data/sheets", exist_ok=True)  # Cria a pasta 'sheets' se não existir
                archive_output = os.path.join("./data/sheets", f"{archive_name}.xlsx")
                df = pd.DataFrame(rows)
                df.to_excel(archive_output, index=False)
                
                print("[X] Arquivo salvo com sucesso!")
                      
            case "2":
                sheet_name = input("Digite o nome da planilha para treinar (sem o .xlsx): ").strip()
                sheet_path = os.path.join("./data/sheets/", f"{sheet_name}.xlsx")
                
                if not os.path.exists(sheet_path):
                    print(f"Erro: Planilha '{sheet_path}' não encontrada!")
                else:
                    df = pd.read_excel(sheet_path)
                    try:
                        classifier.train(df)
                        print("[X] Treinado com sucesso!")
                        classifier.save_model()
                    except Exception as e:
                        print(f"Erro de treinamento: {e}")          
                
            case "3":
                model_path= input("Insira o caminho do modelo (pressione Enter para './data/model/random_forest_trainned.pkl')").strip()
                if model_path == "":
                    model_path = "./data/model/random_forest_trainned.pkl"
                classifier.load_model(model_path)
                
            case "4":
                ### Fazer predict e pintar.
                ### Carregar planilha
                sheet_name = input("Digite o nome da planilha para predizer(sem o .xlsx): ").strip()
                sheet_path = os.path.join("./data/sheets/", f"{sheet_name}.xlsx")
                
                df = pd.read_excel(sheet_path)
                df["label_predict"] = classifier.predict(df)
                
                renderer = MapRenderer()
                img_colored = renderer.render(df, original_img)
                os.makedirs("./mapped", exist_ok=True)
                img_colored.save(f"./mapped/{img_path}")
        
            case "5":
                print("\nEncerrando o programa. Até mais!")
                break
            case _:
                print("\nOpcao invalida!")    


if __name__ == "__main__":
    main()
