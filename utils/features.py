import numpy as np

def extraction_features(img_array_slice):
    """Extrai características da imagem."""
    # 1. Mede o tom medio de vermelho: solos expostos e terras tendem a ter valores altos aqui
    mean_r = np.mean(img_array_slice[:, :, 0])
    # 2. Mede o tom medio de verde: vegetacoes e copas de arvores se destacam com valores altos aqui
    mean_g = np.mean(img_array_slice[:, :, 1])
    # 3. Mede o tom medio de azul: corpos d'agua limpos tendem a refletir mais este comprimento de onda
    mean_b = np.mean(img_array_slice[:, :, 2])
    # 4. Converte o bloco para brilho perceptual (tons de cinza), dando mais peso ao verde conforme a visao humana
    cinza = (
        0.299 * img_array_slice[:, :, 0]
        + 0.587 * img_array_slice[:, :, 1]
        + 0.114 * img_array_slice[:, :, 2]
    )
    # 5. Mede a rugosidade global: agua calma tem desvio perto de zero, florestas com folhas e sombras tem desvio alto
    desvio = np.std(cinza)
    # 6. Mede a diferenca brusca de luz entre cada pixel e seu vizinho da direita (bordas horizontais)
    grad_h = np.mean(np.abs(cinza[:, :-1] - cinza[:, 1:]))
    # 7. Mede a diferenca brusca de luz entre cada pixel e seu vizinho de baixo (bordas verticais)
    grad_v = np.mean(np.abs(cinza[:-1, :] - cinza[1:, :]))
    # 8. Média dos gradientes: resume a frequencia espacial de microtexturas e aspereza do terreno
    gradiente = (grad_h + grad_v) / 2.0

    return [mean_r, mean_g, mean_b, desvio, gradiente]