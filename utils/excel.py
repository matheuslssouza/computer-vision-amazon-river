import os
import pandas as pd

def save_features_to_excel(features_list, archive_output):
    
    """
    Salva uma lista de características em um arquivo Excel.
    """
    # Gera um nome de arquivo único
    df = pd.DataFrame(features_list)
    df.to_excel(archive_output , index=False)