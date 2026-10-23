# Classificação de Câncer de Mama — KNN e SVM

Tech Challenge da Fase 1 da pós-graduação **IA para DEVs (FIAP)**.

O notebook usa o dataset [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) para prever se um tumor é **maligno (M)** ou **benigno (B)**. Ele cobre:

- análise exploratória (valores ausentes, boxplots, histogramas, pairplot e matriz de correlação);
- tratamento dos dados: remove as colunas `_se` e `_worst`, `id` e `Unnamed: 32`, e converte `diagnosis` para 1/0;
- divisão treino/teste estratificada (80/20) e padronização com `StandardScaler`;
- **KNN**: escolha do `k` pela taxa de erro, depois treino e acurácia;
- **SVM** (`LinearSVC`): treino e acurácia;
- importância das variáveis por *permutation importance* (métrica: recall).

O relatório técnico com o pré-processamento, os modelos e os resultados está em [docs/relatorio-tecnico.md](docs/relatorio-tecnico.md) (também em [PDF](docs/relatorio-tecnico.pdf)).

## Estrutura

```
.
├── breast-cancer-knn-svm-random-forest.ipynb   # notebook principal
├── data/
│   └── data.csv                                # dataset (já incluído)
├── docs/
│   ├── relatorio-tecnico.md                    # relatório técnico
│   ├── relatorio-tecnico.pdf
│   └── importancia-variaveis-knn.png           # gráfico usado no relatório
├── requirements.txt
└── README.md
```

## Pré-requisitos

- Python 3.10 ou superior
- `pip`

## Como rodar

### 1. Clonar o repositório

```bash
git clone https://github.com/hugocesarcoelho/fiap-tc-fase1-ai.git
cd fiap-tc-fase1-ai
```

### 2. Criar e ativar um ambiente virtual

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Abrir o notebook

```bash
jupyter notebook breast-cancer-knn-svm-random-forest.ipynb
```

Também dá para abrir o `.ipynb` direto no **VS Code** (extensões Python e Jupyter) e selecionar o kernel do `.venv`.

Depois, execute todas as células (**Run All**). Inicie o Jupyter a partir da raiz do projeto, porque o notebook lê o dataset pelo caminho relativo `data/data.csv`.

## Dataset

O arquivo `data/data.csv` já vem no repositório. Para baixar de novo pelo Kaggle, descomente a primeira célula do notebook, que usa o `kagglehub`:

```python
%pip install kagglehub
import kagglehub
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
```

Em seguida, copie o `data.csv` baixado para a pasta `data/`.
