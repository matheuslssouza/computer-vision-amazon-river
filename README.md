# 🛰️ Visão Computacional em Sensoriamento Remoto: Rio Amazonas

Olá! Este repositório documenta a minha jornada prática de **aprendizado em Visão Computacional e Inteligência Artificial**, explorando os fundamentos matemáticos e algoritmos clássicos de processamento de imagens e texturas.

O objetivo deste projeto é analisar imagens de sensoriamento remoto (focando no **Rio Amazonas**, áreas de vegetação densa e solo exposto), entendendo a mecânica "à mão", pixel por pixel, antes de utilizar bibliotecas de caixa preta.

---

## 🎯 Objetivo de Aprendizado

- Entender a representação de imagens como matrizes tridimensionais $(\text{Altura}, \text{Largura}, \text{Canais RGB})$.
- Aplicar a técnica de **Janela Deslizante (*Sliding Window*)** para varredura do terreno em fatias de $16 \times 16$ pixels.
- Extrair características matemáticas e assinaturas estatísticas de cada bloco sem atalhos mágicos:
  1. **Médias dos Canais de Cores (RGB):** Distinção da resposta espectral de água, folhas e solo.
  2. **Conversão Perceptual para Escala de Cinza:** Ponderação luminosa pelo padrão ITU-R BT.601 ($0.299R + 0.587G + 0.114B$).
  3. **Desvio Padrão ($\sigma$):** Medição da rugosidade global da textura (água calma possui dispersão próxima de zero; florestas com copas e sombras possuem dispersão altíssima).
  4. **Gradiente Espacial Local ($G_h, G_v$):** Medição da frequência de bordas e variações bruscas de luminosidade entre pixels vizinhos imediatos.
- Integrar com classificadores supervisionados de Machine Learning (**Random Forest** e **SVM**) para mapeamento e rotulagem temática.

---

## 📂 Estrutura do Projeto

```
computer-vision-amazon-river/
├── .gitignore               # Ignora arquivos temporários e .venv
├── main.py                  # Script principal com a extração das métricas
├── ROTEIRO_CONCEITUAL.md    # Guia detalhado de arquitetura e lógica de cada etapa
├── teste.jpg                # Imagem de satélite para experimentação
└── README.md                # Este documento
```

---

## 🚀 Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/matheuslssouza/computer-vision-amazon-river.git
   cd computer-vision-amazon-river
   ```

2. Crie e ative um ambiente virtual (opcional):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Instale as dependências mínimas:
   ```bash
   pip install numpy pillow scikit-learn
   ```

4. Execute o script principal:
   ```bash
   python main.py
   ```

---

*Projeto desenvolvido para fins de estudo e experimentação prática em IA e Visão Computacional.*
