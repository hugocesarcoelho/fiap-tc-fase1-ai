# Tech Challenge — Classificação de Câncer de Mama

**FIAP · Pós-Tech IA para Devs · Fase 1**

Diagnóstico maligno × benigno com KNN, SVM e Random Forest

**Integrantes**

- Hugo César Coelho de Oliveira — RM ________

Outubro de 2026

## 1. Links da entrega

| | |
| --- | --- |
| **Repositório Git** | [https://github.com/hugocesarcoelho/fiap-tc-fase1-ai](https://github.com/hugocesarcoelho/fiap-tc-fase1-ai) |
| **Vídeo de demonstração** | [https://youtu.be/vv8gxPOWYq0](https://youtu.be/vv8gxPOWYq0) |
| **Dataset** | Incluído em `data/data.csv` · original: [Kaggle — Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) |

## 2. Conteúdo do repositório

| Entregável | Onde está |
| --- | --- |
| Código-fonte completo | `breast-cancer-knn-svm-random-forest.ipynb` — análise exploratória, pré-processamento, treino e avaliação dos três modelos |
| Dockerfile e README.md com instruções de execução | `Dockerfile` — imagem Python 3.12 com as dependências, o notebook e o dataset, servindo o Jupyter na porta 8888. `README.md` — execução com Docker ou com ambiente virtual (`requirements.txt` com versões fixadas) |
| Dataset | `data/data.csv` (569 exames, 32 colunas) |
| Resultados (prints, gráficos e análises) | Saídas salvas no notebook e imagens em `docs/resultados/`; seção 4 deste documento |
| Relatório técnico | `docs/relatorio-tecnico.md` e `docs/relatorio-tecnico.pdf`; seção 5 deste documento |

## 3. Como executar

### Com Docker

```bash
git clone https://github.com/hugocesarcoelho/fiap-tc-fase1-ai.git
cd fiap-tc-fase1-ai
docker build -t fiap-tc-fase1-ai .
docker run --rm -p 8888:8888 fiap-tc-fase1-ai
```

Abra **http://localhost:8888**, entre em `breast-cancer-knn-svm-random-forest.ipynb` e execute todas as células (**Run All**).

### Sem Docker (Python 3.11+)

```bash
git clone https://github.com/hugocesarcoelho/fiap-tc-fase1-ai.git
cd fiap-tc-fase1-ai
python -m venv .venv
.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook breast-cancer-knn-svm-random-forest.ipynb
```

Execute todas as células (**Run All**). O notebook lê o dataset pelo caminho relativo `data/data.csv`, então deve ser aberto a partir da raiz do projeto.

## 4. Resultados obtidos

### 4.1 Desempenho dos modelos (conjunto de teste, 114 exames)

| Métrica | KNN (k = 5) | SVM linear | Random Forest |
| --- | --- | --- | --- |
| Acurácia | **92,1%** | **92,1%** | 88,6% |
| Recall de malignos | **85,7%** (36 de 42) | 83,3% (35 de 42) | 76,2% (32 de 42) |
| Precisão de malignos | 92,3% | **94,6%** | 91,4% |
| F1 de malignos | **0,889** | 0,886 | 0,831 |
| Falsos negativos | **6** | 7 | 10 |
| Falsos positivos | 3 | **2** | 3 |

Matrizes de confusão e métricas por classe, da célula de comparação do notebook, lado a lado (classe 1 = maligno):

```text
KNN                         SVM                         Random Forest
[[69  3]                    [[70  2]                    [[69  3]
 [ 6 36]]                    [ 7 35]]                    [10 32]]
   precision recall  f1        precision recall  f1        precision recall  f1
0    0.920  0.958  0.939     0    0.909  0.972  0.940     0    0.873  0.958  0.914
1    0.923  0.857  0.889     1    0.946  0.833  0.886     1    0.914  0.762  0.831
accuracy           0.921     accuracy           0.921     accuracy           0.886
```

### 4.2 Escolha do k e importância das variáveis

![Figura 1](resultados/cell21_0.png)

*Figura 1. Taxa de erro do KNN no teste para k de 1 a 9. Escolhido k = 5.*

![Figura 2](resultados/cell28_0.png)

*Figura 2. Importância das variáveis no KNN por permutação (queda do recall de malignos).*

![Figura 3](resultados/cell27_0.png)

*Figura 3. Importância das variáveis no Random Forest (redução de impureza).*

### 4.3 Análise exploratória

![Figura 4](resultados/cell03_0.png)

*Figura 4. Matriz de valores ausentes (missingno): nenhuma medida tem valores faltantes; só a coluna `Unnamed: 32` é vazia.*

![Figura 5](resultados/cell05_0.png)

*Figura 5. `area_mean` por diagnóstico (B = benigno, M = maligno).*

![Figura 6](resultados/cell06_0.png)

*Figura 6. `concavity_mean` por diagnóstico.*

![Figura 7](resultados/cell07_0.png)

*Figura 7. `smoothness_mean` por diagnóstico.*

![Figura 8](resultados/cell08_0.png)

*Figura 8. Histograma de `area_mean` por diagnóstico.*

![Figura 9](resultados/cell09_0.png)

*Figura 9. Swarmplot da área média do tumor por diagnóstico.*

![Figura 10](resultados/cell10_0.png)

*Figura 10. Pairplot de área, simetria e concavidade: área × concavidade separa bem as classes.*

![Figura 11](resultados/cell14_0.png)

*Figura 11. Matriz de correlação das 10 variáveis médias e do diagnóstico.*

## 5. Relatório técnico

As figuras citadas estão na seção 4.

### Introdução

O KNN (k = 5) e o SVM linear acertaram 92,1% dos casos de teste (105 de 114 tumores); o Random Forest acertou 88,6%. O KNN é o recomendado, por encontrar mais casos de câncer: 36 de 42 malignos.

O trabalho é o Tech Challenge da Fase 1 da pós-graduação IA para DEVs (FIAP). O objetivo é apoiar o diagnóstico de câncer de mama com um classificador binário: maligno (M) ou benigno (B).

Os dados vêm do [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). São 569 exames, 357 benignos (62,7%) e 212 malignos (37,3%). Cada exame tem 30 medidas do núcleo celular, extraídas de imagens de punção aspirativa por agulha fina. O código está no [repositório do projeto](https://github.com/hugocesarcoelho/fiap-tc-fase1-ai).

### Estratégias de pré-processamento

Os dados vieram limpos: só foi preciso escolher colunas, codificar o alvo, dividir treino e teste e padronizar a escala.

1. **Seleção de variáveis.** Cada medida aparece em três versões: média (`_mean`), erro padrão (`_se`) e pior valor (`_worst`). Mantivemos só as 10 médias. As versões `_worst` são muito correlacionadas com as médias, e as `_se` medem variação, não tamanho ou forma. Com menos colunas, o KNN sofre menos com dimensão alta e o modelo fica mais fácil de explicar.
2. **Valores ausentes.** A matriz do `missingno` e o `isnull().sum()` mostraram zero ausentes nas medidas. A única coluna vazia, `Unnamed: 32` (569 nulos), é um artefato do CSV e foi removida, junto com o `id`, que não tem valor preditivo.
3. **Codificação do alvo.** `diagnosis` virou numérica: maligno = 1, benigno = 0. Assim a classe positiva é a doença, e o recall mede quantos casos de câncer o modelo encontra.
4. **Divisão treino/teste.** 80% para treino (455 exames) e 20% para teste (114), com `stratify=y` e `random_state=7`. A estratificação mantém a proporção de cerca de 37% de malignos nas duas partes: 170 de 455 no treino e 42 de 114 no teste.
5. **Padronização.** `StandardScaler` deixa cada variável com média 0 e desvio 1. Sem isso, `area_mean` (centenas a milhares) dominaria a distância do KNN sobre `smoothness_mean` (cerca de 0,1). O scaler foi ajustado só no treino e depois aplicado ao teste, para evitar vazamento de dados.

A análise exploratória (boxplots, histograma, swarmplot e pairplot) orientou essas escolhas. Ela mostrou que tumores malignos têm área e concavidade maiores, enquanto a simetria quase não separa os grupos.

### Modelos usados e por quê

Comparamos três classificadores de natureza diferente: um baseado em vizinhança (KNN), um em fronteira linear (SVM) e um conjunto de árvores de decisão (Random Forest). KNN e SVM usaram os dados padronizados. O Random Forest usou os dados originais, porque árvores não dependem da escala.

| Modelo | Configuração | Por que foi escolhido |
| --- | --- | --- |
| KNN (`KNeighborsClassifier`) | k = 5, distância euclidiana | Simples e sem suposição sobre a forma dos dados. Classifica pelo diagnóstico dos 5 exames mais parecidos, o que é fácil de explicar. Funciona bem com poucas variáveis contínuas já padronizadas. |
| SVM linear (`LinearSVC`) | C = 1 | O pairplot mostrou que as classes se separam razoavelmente por uma fronteira (área e concavidade altas = maligno). O SVM busca a fronteira de maior margem e costuma generalizar bem em bases pequenas como esta (455 exames de treino). |
| Random Forest (`RandomForestClassifier`) | 100 árvores, `random_state=42` | Capta relações não lineares e interações entre variáveis sem exigir padronização. Por combinar muitas árvores, é robusto a ruído e tem uma importância de variáveis própria, que serve de contraponto à do KNN. |

**Escolha do k.** Testamos k de 1 a 9 e medimos a taxa de erro no teste. O erro cai de 9,6% (k = 1) para 7,9% em k = 3, 4 e 5, sobe para 8,8% em k = 6 e volta a 7,9% de k = 7 a 9. Escolhemos k = 5, dentro da faixa de erro mínimo: um k ímpar evita empate entre as duas classes, e um k maior reduz a sensibilidade a exames atípicos.

Além do k do KNN, nenhum hiperparâmetro foi ajustado: o `C` do SVM e o número de árvores do Random Forest ficaram nos valores padrão.

### Resultados e interpretação

KNN e SVM empataram em acurácia (92,1%), mas o KNN encontrou um câncer a mais: recall de 85,7% contra 83,3% do SVM. O Random Forest ficou em último nas duas métricas, com 88,6% de acurácia e 76,2% de recall.

| Métrica (teste, 114 exames) | KNN (k = 5) | SVM linear | Random Forest |
| --- | --- | --- | --- |
| Acurácia | 92,1% | 92,1% | 88,6% |
| Recall de malignos | 85,7% (36 de 42) | 83,3% (35 de 42) | 76,2% (32 de 42) |
| Precisão de malignos | 92,3% | 94,6% | 91,4% |
| F1 de malignos | 0,889 | 0,886 | 0,831 |
| Falsos negativos (maligno previsto como benigno) | 6 | 7 | 10 |
| Falsos positivos (benigno previsto como maligno) | 3 | 2 | 3 |

Todos os números vêm do notebook: a acurácia de cada modelo e, na célula de comparação, a matriz de confusão e o relatório de classificação dos três.

**Por que o recall importa mais que a acurácia.** Num exame de triagem, um falso negativo manda para casa uma paciente com câncer; um falso positivo gera um exame a mais. Por isso o KNN é o modelo preferido aqui: tem a mesma acurácia do SVM e o maior recall dos três. Ainda assim, 6 de 42 malignos escapam, e esse é o principal ponto a melhorar.

**Por que o Random Forest ficou atrás.** Com 455 exames de treino e 10 variáveis bem correlacionadas, ele não teve vantagem sobre fronteiras mais simples. O modelo concentrou a importância em medidas de contorno e tamanho: `concave points_mean` (30%), `concavity_mean` (18%) e `perimeter_mean` (17%). Sem ajuste de hiperparâmetros nem do limiar de decisão, foi o mais conservador ao prever maligno: 10 falsos negativos, contra 6 do KNN.

**Importância das variáveis.** No KNN, a importância foi medida por permutação: embaralha-se uma variável e mede-se quanto o recall cai. Textura e forma da célula (suavidade, textura, simetria) pesam mais que tamanho.

Isso parece contradizer a correlação: `concave points_mean` (0,78), `perimeter_mean` (0,74) e `radius_mean` (0,73) são as mais correlacionadas com o diagnóstico. A explicação é a redundância. Raio, perímetro e área medem quase a mesma coisa, então embaralhar uma delas pouco muda, porque as outras compensam. Os desvios são grandes: em 5 das 10 variáveis a barra cruza o zero, e nenhuma sozinha é decisiva.

**Análise exploratória.** Os gráficos confirmam o que os modelos aprenderam. Malignos têm `area_mean` e `concavity_mean` claramente maiores, e o par área × concavidade separa bem os grupos no pairplot. Já `symmetry_mean` tem forte sobreposição entre as classes.

### Conclusão e próximos passos

Com só 10 medidas médias e padronização, o KNN (k = 5) chega a 92,1% de acurácia e 85,7% de recall. É o modelo recomendado entre os três testados; o Random Forest, sem ajuste, ficou em 88,6% e 76,2%. Por deixar escapar 6 de 42 malignos, o KNN serve como apoio à triagem, não como substituto do diagnóstico médico.

- Usar validação cruzada estratificada: com 114 exames de teste, cada erro move a acurácia em 0,9 ponto.
- Ajustar hiperparâmetros (k e pesos do KNN, `C` do SVM) com `GridSearchCV`, otimizando o recall.
- Testar as variáveis `_worst` e `_se` hoje descartadas.
- Ajustar o Random Forest (profundidade, número de árvores, `class_weight`) e o limiar de decisão para reduzir os falsos negativos.
