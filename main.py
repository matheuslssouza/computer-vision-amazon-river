import numpy as np
import os
from PIL import Image
import utils.excel as Excel
import utils.images as Images
import utils.features as Features
import utils.trainning as Tranning

"""
Abrir imagem e colocar na memoria em array numpy
"""

def main():
    SLICE = 16  # Altura da fatia em pixels
    img_array = Images.open_image("./teste.jpg")
    linhas_tabela = []
    id_sequencial = 0

    # Processa a imagem em fatias de 16 pixels de altura
    for i in range(0, img_array.shape[0], SLICE):
        for j in range(0, img_array.shape[1], SLICE):
            img_array_slice = img_array[i : i + SLICE, j : j + SLICE]

            if (
                img_array_slice.shape[0] < SLICE
                or img_array_slice.shape[1] < SLICE
            ):
                continue  # Ignora fatias menores que 16x16 pixels

            id_sequencial += 1

            features_result = Features.extraction_features(img_array_slice)
            linhas_tabela.append(
                {
                    "id": id_sequencial,
                    "coordenada_x": j,
                    "coordenada_y": i,
                    "archive": f"bloco_{id_sequencial:04d}",
                    "Mean Red": features_result[0],
                    "Mean Green": features_result[1],
                    "Mean Blue": features_result[2],
                    "DesvPad": features_result[3],
                    "Gradient": features_result[4],
                    "tipo": "",  # Campo para classificação manual posterior
                }
            )
            print(f"Features for slice ({i}, {j}): {features_result}")

            Images.save_image(img_array_slice, f"bloco_{id_sequencial:04d}")

    archive_name = input("Digite o nome do arquivo Excel: ").strip()
    os.makedirs("./data/sheets", exist_ok=True)  # Cria a pasta 'sheets' se não existir
    archive_output = os.path.join("./data/sheets", f"{archive_name}.xlsx")


    # Excel.save_features_to_excel(linhas_tabela, archive_output)
    
    Tranning.trainning_model(archive_output, SLICE)


if __name__ == "__main__":
    main()
