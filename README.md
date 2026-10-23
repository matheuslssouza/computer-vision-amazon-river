# 🛰️ Visão Computacional em Sensoriamento Remoto: Rio Amazonas

Este repositório documenta um projeto prático de **Visão Computacional e Machine Learning clássico**, combinando engenharia manual de características (*Feature Engineering*) com o algoritmo **Random Forest** para classificação do uso e cobertura do solo em imagens de sensoriamento remoto (focando no **Rio Amazonas**, áreas de vegetação densa e corpos d'água).

Diferente de abordagens modernas com redes neurais de "caixa-preta", o objetivo deste projeto é entender a mecânica matemática e estatística, pixel por pixel, extraindo descritores físicos de cor e textura para alimentar um comitê de árvores de decisão.

---

## 🧠 Como Funciona o Pipeline de Machine Learning

O fluxo de processamento e classificação opera em 4 etapas interdependentes:

```mermaid
flowchart TD
    A["1. Imagem de Satélite Crua\n(Matriz RGB)"] --> B["2. Janela Deslizante (16x16)\n(SlidingWindow)"]
    B --> C["3. Extrator de Características\n(FeatureExtractor)"]
    C --> D["4. Planilha Estruturada (.xlsx)\n[Médias RGB, DesvPad, Gradiente]"]
    D --> E["5. Rotulagem Humana (Ground Truth)\n(Anotação manual de 'rio' e 'mata')"]
    E --> F["6. Treinamento da Random Forest\n(Classificador Supervisionado)"]
    F --> G["7. Inferência em Lote (Predict)\n(Classificação de todos os blocos)"]
    G --> H["8. Renderização Translúcida\n(Alpha Blending do Mapa Final)"]
```

### 1. Engenharia de Características (*Feature Engineering*)
Cada recorte de $16 \times 16$ pixels é resumido em um vetor matemático composto por **5 descritores estatísticos**:
* **Médias dos Canais de Cores ($R, G, B$):** Identifica a assinatura espectral básica (a água absorve luz e reflete mais azul/ciano; florestas refletem verde; solos expostos refletem tons terrosos).
* **Luminância Perceptual (Escala de Cinza):** Ponderação luminosa pelo padrão **ITU-R BT.601** ($0.299R + 0.587G + 0.114B$).
* **Desvio Padrão ($\sigma$):** Mede a **rugosidade global** da textura (a superfície da água calma tem dispersão próxima de zero; copas de árvores com folhas e sombras possuem alta variância de luz).
* **Gradiente Espacial Local ($G$):** Mede a **frequência espacial de bordas** calculando a diferença absoluta de brilho entre pixels vizinhos horizontais e verticais:
  $$G_h = \frac{1}{N} \sum |I_{x, y} - I_{x+1, y}| \quad , \quad G_v = \frac{1}{N} \sum |I_{x, y} - I_{x, y+1}| \quad , \quad G = \frac{G_h + G_v}{2}$$

### 2. Classificação Supervisionada: Random Forest
* O modelo **Random Forest** cria um conjunto de dezenas de árvores de decisão independentes treinadas sobre os vetores de características.
* As árvores aprendem regras de separação estatística (ex: *"Se o desvio padrão for $> 25.0$ e a resposta verde for predominante, classifique como mata"*).
* **Serialização de Pesos:** O modelo treinado é salvo no disco via `joblib` (`.pkl`), permitindo classificar novas imagens instantaneamente sem precisar repetir o processo de anotação e treinamento.

### 3. Síntese do Mapa Temático (*Alpha Blending*)
Em vez de desenhar blocos sólidos opacos, o sistema aplica uma **fusão translúcida** (50% cor temática + 50% foto original):
$$\text{Pixel Final} = (\text{Cor da Classe} \times \alpha) + (\text{Foto Original} \times (1 - \alpha))$$
* **Rio:** Azul `[0, 0, 255]`
* **Mata:** Verde `[0, 255, 0]`

---

## 📂 Arquitetura do Projeto (Orientada a Objetos)

O projeto é estruturado em módulos com responsabilidades únicas e desacopladas:

```text
computer-vision-amazon-river/
│
├── core/
│   ├── models.py         # PatchMetrics: Dataclass que modela o bloco e suas métricas
│   ├── window.py         # SlidingWindow: Gerador (Iterator/Yield) que fatia a imagem
│   ├── features.py       # FeatureExtractor: Métodos matemáticos de extração de textura
│   ├── classifier.py     # Classifier: Random Forest com fit, predict, save e load
│   └── renderer.py       # MapRenderer: Aplica as predições e o alpha blending na imagem
│
├── data/
│   ├── sheets/           # Planilhas Excel com as métricas geradas (.xlsx)
│   ├── slices/           # Recortes individuais dos blocos em PNG
│   └── model/            # Modelos treinados salvos em formato binário (.pkl)
│
├── mapped/               # Diretório de saída dos mapas finais pintados
├── main.py               # Interface de terminal interativa (menu de opções)
├── teste.jpg             # Imagem de satélite para processamento
├── requirements.txt      # Dependências do projeto
└── README.md             # Documentação do projeto
```

---

## 🚀 Como Executar e Usar

### 1. Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/matheuslssouza/computer-vision-amazon-river.git
   cd computer-vision-amazon-river
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
   *(Caso não use o requirements.txt: `pip install numpy pillow pandas scikit-learn openpyxl joblib`)*

---

### 2. Guia de Uso Passo a Passo

Execute o script principal:
```bash
python main.py
```

O terminal exibirá o painel interativo:

```text
===================================
    PAINEL DE CLASSIFICAÇÃO
===================================
1 - Extrair caracteristicas de uma imagem e salvar dados da extracao
2 - Treinar modelo
3 - Abrir (carregar) modelo existente
4 - Identificar e mapear
5 - Sair
===================================
```

Siga a sequência abaixo para o pipeline completo:

#### Passo 1: Extrair Características (Opção 1)
* O script percorre a imagem `teste.jpg`, fatia em blocos de $16 \times 16$ pixels, extrai as métricas de cada bloco e salva em uma planilha Excel (ex: `teste.xlsx` dentro de `data/sheets/`).
* Os recortes de cada bloco também são salvos em `data/slices/`.

#### Passo 2: Rotulagem Humana (No Excel)
* Abra a planilha gerada (`data/sheets/teste.xlsx`) no Excel.
* Localize a coluna **`label`**.
* Inspecione alguns recortes da pasta `data/slices/` e anote algumas linhas de referência (ex: digite `"rio"` em 10 linhas e `"mata"` em 10 linhas). Deixe o restante das linhas em branco.
* Salve e feche a planilha.

#### Passo 3: Treinar o Modelo (Opção 2)
* Selecione a opção **2** e informe o nome da planilha.
* O algoritmo **Random Forest** será treinado com base nas suas anotações e os pesos aprendidos serão salvos automaticamente em `data/model/random_forest_trainned.pkl`.

#### Passo 4: Identificar e Mapear (Opção 4)
* Selecione a opção **4**.
* O classificador prevê a categoria de **todos** os blocos da foto inteira em lote.
* O `MapRenderer` gera o mapa temático colorido sobreposto e salva o resultado final na pasta `mapped/`.

#### Passo Alternativo: Reutilizar Modelo sem Treinar (Opção 3)
* Caso já possua um modelo previamente treinado, use a opção **3** para carregar os pesos `.pkl` do disco e pule direto para a opção **4**.

---

*Projeto desenvolvido para fins de estudo e experimentação prática em IA e Sensoriamento Remoto.*
