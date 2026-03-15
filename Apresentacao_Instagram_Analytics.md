# 📊 Apresentação — Instagram Analytics
## Explicação Completa do Notebook | Público Técnico e Leigo

---

## 🎯 Objetivo do Projeto

Analisar os dados de performance de posts no Instagram para entender o que gera mais alcance, engajamento e novos seguidores.

**Dataset:** 119 posts com métricas como impressões, likes, saves, comentários, compartilhamentos e origem do tráfego.

---

## Ordem de Execução do Notebook

1. Configuração do Ambiente
2. Leitura e Inspeção Inicial
3. Qualidade dos Dados
4. Engenharia de Features
5. Normalização e One-Hot Encoding
6. Estatísticas Descritivas
7. Detecção de Outliers
8. Análise de Distribuição
9. Correlações
10. Análise de Texto
11. Insights e Conclusões

---

## 1. ⚙️ Configuração do Ambiente

### Para o Data Scientist
Instalação das bibliotecas `wordcloud` e `missingno`. Importação de `pandas`, `numpy`, `matplotlib`, `seaborn` e `sklearn`. Configuração do estilo visual padrão para todos os gráficos da análise.

### Para o público leigo
Antes de começar a análise, preparamos todas as ferramentas necessárias — como montar um kit de instrumentos antes de iniciar um trabalho. Cada ferramenta tem uma função específica: uma para criar gráficos, outra para analisar textos, outra para detectar problemas nos dados.

---

## 2. 📂 Leitura e Inspeção Inicial

### Para o Data Scientist
```python
df = pd.read_csv('/content/Instagram_data.csv', encoding='latin1')
df = df.replace('amankharwal', 'suasaladatech', regex=True)
```
Encoding `latin1` necessário devido a caracteres especiais nas colunas de texto. Substituição de `amankharwal` por `suasaladatech` aplicada imediatamente após a leitura via `regex=True` — garante que a troca ocorre em qualquer parte do texto das colunas Caption e Hashtags.

**Resultado:** 119 linhas × 13 colunas
- 11 colunas numéricas (int64): métricas de performance
- 2 colunas texto (object): Caption e Hashtags
- 117 ocorrências de `suasaladatech` confirmadas

### Para o público leigo
Carregamos o arquivo com os dados dos posts do Instagram — como abrir uma planilha Excel com 119 linhas (uma por post) e 13 colunas (informações de cada post). Também fizemos uma operação de "localizar e substituir" para trocar o nome do autor original pelo nome correto `suasaladatech` em todos os textos. A substituição foi confirmada em 117 ocorrências.

---

## 3. 🔍 Qualidade dos Dados

### 3.1 Valores Nulos

#### Para o Data Scientist
Verificação com `df.isnull().sum()` e visualização com `missingno.matrix()`.

**Resultado:** ✅ Zero valores nulos — dataset 100% preenchido em todas as 13 colunas.

#### Para o público leigo
Verificamos se havia campos em branco nos dados — como uma ficha cadastral com informações faltando. Resultado ótimo: nenhum dado faltando em nenhum dos 119 posts.

---

### 3.2 Duplicatas

#### Para o Data Scientist
Análise em três níveis:

| Critério | Resultado |
|---|---|
| `df.duplicated()` — todas as colunas | **17 linhas removidas** |
| `subset` colunas numéricas | **17 linhas** |
| `subset=['Caption']` | **29 ocorrências** |

Os 17 posts duplicados são cópias exatas das linhas 15–29 (repetidas nas linhas 82–98), indicando duplicação em bloco do dataset. Investigação adicional revelou **27 Captions distintos** que se repetem, envolvendo **56 posts** — conteúdo reciclado ou reposts intencionais.

Após `drop_duplicates()`: **102 linhas restantes**.

#### Para o público leigo
Verificamos se algum post foi registrado mais de uma vez por engano — como ter linhas repetidas numa planilha. Encontramos 17 posts completamente idênticos e os removemos. Também descobrimos que 27 textos de legenda foram usados em mais de um post — o mesmo conteúdo foi publicado novamente em outro momento. Após a limpeza, ficamos com 102 posts únicos.

---

### 3.3 Verificação de Tipos

#### Para o Data Scientist
```
Numéricas (11): Impressions, From Home, From Hashtags, From Explore,
                From Other, Saves, Comments, Shares, Likes,
                Profile Visits, Follows

Texto (2): Caption, Hashtags
```
Tipos corretos — não foi necessária conversão de dtype.

#### Para o público leigo
Confirmamos que números estavam em colunas numéricas e textos em colunas de texto — como verificar se uma planilha está formatada corretamente. Tudo certo, sem ajustes necessários.

---

## 4. 🛠️ Engenharia de Features

### Para o Data Scientist
Criação de 6 novas variáveis derivadas a partir das colunas existentes:

```python
engagement_rate    = (Likes + Comments + Shares + Saves) / Impressions × 100
conversion_rate    = Follows / Profile Visits × 100
main_source        = idxmax(['From Home','From Hashtags','From Explore','From Other'])
impressions_category = pd.cut(bins=[0,3000,5000,10000,∞],
                               labels=['Baixo','Médio','Alto','Viral'])
caption_length     = len(Caption)
num_hashtags       = count(# in Hashtags)
```

**Resultados:**
- Engagement Rate médio: ~5-6%
- Conversion Rate médio: ~20%
- Main Source: `From Home` dominante (~70% dos posts)
- Distribuição: maioria em Médio e Alto

### Para o público leigo
Criamos novas informações a partir das existentes — como calcular a média de uma turma a partir das notas individuais. As mais importantes:

**Taxa de Engajamento:** de cada 100 pessoas que viram o post, ~5 a 6 interagiram (curtindo, comentando, salvando ou compartilhando). Isso é considerado bom para o Instagram.

**Taxa de Conversão:** de cada 100 pessoas que visitaram o perfil, ~20 viraram seguidores — uma taxa de conversão alta, indicando que o conteúdo convence quem chega ao perfil.

**Origem principal:** a maioria das visualizações vem de quem já segue a conta, o que indica uma audiência fiel.

**Categorias de alcance criadas:**
- Baixo → menos de 3.000 impressões
- Médio → 3.000 a 5.000
- Alto → 5.000 a 10.000
- Viral → acima de 10.000

---

## 5. 🔢 Normalização e One-Hot Encoding

### Para o Data Scientist

**Min-Max Scaler** — intervalo [0, 1]:
```python
valor_normalizado = (valor - mínimo) / (máximo - mínimo)
```
Verificação: min=0.00 e max=1.00 confirmados para todas as 15 colunas numéricas.

**Standard Scaler (Z-score)** — média=0, desvio=1:
```python
z = (valor - média) / desvio_padrão
```
Verificação: mean≈0.00 e std=1.00 confirmados para todas as colunas.

**One-Hot Encoding — `main_source` (variável nominal sem ordem):**
```
From Home     → source_From Home     (1 ou 0)
From Hashtags → source_From Hashtags (1 ou 0)
From Explore  → source_From Explore  (1 ou 0)
```

**Label Encoding — `impressions_category` (variável ordinal com ordem):**
```
Baixo=0 | Medio=1 | Alto=2 | Viral=3
```

**Dataset final:** 102 linhas × 19 colunas — pronto para modelagem.

### Para o público leigo
Antes de aplicar algoritmos de Machine Learning, precisamos colocar todos os dados na mesma escala e converter textos em números — como converter temperaturas de Celsius para Fahrenheit antes de comparar.

**Por que normalizar?** Impressions vai de 1.941 a 36.919, enquanto Comments vai de 0 a 19. Essa diferença de escala confunde os algoritmos. Normalizamos tudo para ficar entre 0 e 1.

**Por que fazer encoding?** Computadores não entendem texto como "From Home". Criamos colunas de 0s e 1s: se a origem principal foi "From Home", a coluna `source_From Home` recebe 1 e as outras ficam com 0.

**Resultado:** dataset com 19 colunas, todas numéricas e na mesma escala — pronto para Machine Learning.

---

## 6. 📈 Estatísticas Descritivas

### Para o Data Scientist
Métricas calculadas após limpeza e criação de features: count, mean, std, min, quartis, max, assimetria (skewness) e curtose.

**Principais achados (base: 102 posts):**

| Métrica | Média | Mediana | Desvio Padrão | Assimetria | Curtose |
|---|---|---|---|---|---|
| Impressions | 5.920 | 4.344 | 5.140 | 3.98 | 19.42 |
| Likes | 177 | 158 | 85 | 1.72 | 4.00 |
| Follows | 23 | 8 | 44 | 3.75 | 15.52 |
| Profile Visits | 55 | 24 | 93 | 3.89 | 16.92 |
| Comments | 6 | 6 | 3 | 0.76 | 1.89 |
| From Home | 2.497 | 2.216 | 1.588 | 5.40 | 33.48 |
| From Other | 185 | 75 | 309 | 5.06 | 34.23 |

Alta assimetria positiva em Impressions (3.98), Follows (3.75) e Profile Visits (3.89) indica distribuições com cauda longa à direita. Curtose elevada em From Home (33.48) e From Other (34.23) aponta outliers extremos. Comments é a variável mais próxima de uma distribuição normal (skewness=0.76, curtose=1.89).

### Para o público leigo
Calculamos os resumos estatísticos de cada métrica — como tirar a média, mínimo e máximo das notas de uma turma.

Os números mais importantes:
- **Impressions médias:** 5.920 visualizações por post
- **Likes médios:** 177 por post
- **Novos seguidores médios:** 23 por post
- **Comentários médios:** 6 por post

A diferença entre média e mediana nas Impressions (5.920 vs 4.344) revela que poucos posts virais "puxam" a média para cima — a maioria dos posts fica em torno de 4.344 visualizações.

---

## 7. 🚨 Detecção de Outliers

### Para o Data Scientist
Método IQR: outlier quando valor < `Q1 - 1.5×IQR` ou > `Q3 + 1.5×IQR`.

**Resultado por coluna:**

| Coluna | Limite Sup | Outliers | % |
|---|---|---|---|
| From Explore | 1.553 | 14 | 13.73% |
| Follows | 39 | 13 | 12.75% |
| Impressions | 10.407 | 11 | 10.78% |
| Profile Visits | 90 | 11 | 10.78% |
| From Other | 486 | 10 | 9.80% |
| Saves | 328 | 9 | 8.82% |
| Shares | 28 | 3 | 2.94% |
| Comments | 14 | 2 | 1.96% |

Os 11 outliers de Impressions são posts com mais de 10.407 visualizações. O post mais extremo (linha 118) teve 36.919 impressões, 611 visitas ao perfil e 228 novos seguidores — performance excepcional legítima, não erro de dado.

### Para o público leigo
Outliers são valores muito fora do padrão — como um aluno que tira 10 numa turma onde a média é 6. Encontramos posts com performance muito acima do normal.

O post mais viral teve 36.919 visualizações — quase 9x acima da média. Não é erro: é um post que realmente "bombou", gerando também 611 visitas ao perfil e 228 novos seguidores num único post. Esses casos excepcionais são os mais valiosos para entender o que funciona.

---

## 8. 📉 Análise de Distribuição

### Para o Data Scientist
Histogramas com KDE para todas as 11 variáveis numéricas. A maioria apresenta assimetria positiva (right-skewed), confirmando os valores de skewness da seção anterior. Comments é a única variável com distribuição próxima da normal.

### Para o público leigo
Visualizamos como os dados se distribuem em cada métrica — como um gráfico de notas que mostra quantos alunos tiraram cada nota. A maioria das métricas tem muitos posts com valores baixos e poucos com valores muito altos. Isso é normal em redes sociais: a maioria dos posts tem performance mediana e apenas alguns viralizam.

---

## 9. 🔗 Correlações

### Para o Data Scientist
Matriz de correlação de Pearson. Principais correlações com Impressions:

- **From Explore:** correlação alta positiva — é o maior driver de alcance
- **From Hashtags:** correlação moderada positiva
- **Likes e Follows:** correlação moderada positiva
- **Comments:** correlação baixa

Scatter plots com linha de regressão confirmam relação linear positiva em Impressions×Likes, Impressions×Follows e Likes×Saves.

### Para o público leigo
Analisamos quais fatores "andam juntos" — se quando um sobe, o outro também sobe.

Os achados mais importantes:
- **Aparecer no Explore = muito mais visualizações** — a aba Explorar é o maior impulsionador de alcance do Instagram
- **Mais Likes = mais Follows** — posts que as pessoas curtem tendem a gerar novos seguidores
- **Mais Likes = mais Saves** — conteúdo curtido também é salvo para rever depois

---

## 10. 📝 Análise de Texto

### Para o Data Scientist
**Captions:** comprimento médio de 180 caracteres (std=128), variando de 44 a 784. Scatter de `caption_length × Impressions` não evidencia correlação forte — o tamanho do texto não determina o alcance.

**Hashtags:** 163 hashtags únicas identificadas. Top 5:

| Hashtag | Frequência |
|---|---|
| #suasaladatech | 100 |
| #thecleverprogrammer | 100 |
| #python | 93 |
| #pythonprogramming | 84 |
| #datascience | 79 |

WordCloud confirma dominância de hashtags de Data Science e Python.

### Para o público leigo
Analisamos os textos e as hashtags dos posts.

**Sobre os textos:** os posts têm em média 180 caracteres — equivalente a um tweet longo. Textos maiores ou menores não influenciam o alcance de forma significativa.

**Sobre as hashtags:** foram usadas 163 hashtags diferentes. As mais usadas são `#suasaladatech`, `#python` e `#datascience` — mostrando um nicho de conteúdo bem definido. A nuvem de palavras visualiza quais são as mais frequentes de forma intuitiva.

---

## 11. 💡 Insights e Conclusões

### Resumo do Dataset

| Indicador | Valor |
|---|---|
| Posts analisados | 102 (após limpeza) |
| Valores nulos | 0 |
| Duplicatas removidas | 17 |
| Captions reciclados | 27 |
| Hashtags únicas | 163 |
| Colunas finais | 19 |

### Métricas Médias

| Métrica | Valor |
|---|---|
| Impressions | 5.920 |
| Likes | 177 |
| Saves | 157 |
| Comments | 6 |
| Shares | 9 |
| Profile Visits | 55 |
| Follows | 23 |
| Engagement Rate | ~5-6% |
| Conversion Rate | ~20% |

### Principais Insights

| Insight | O que significa |
|---|---|
| Explore é o maior driver de Impressions | Aparecer na aba Explorar é estratégico para crescimento |
| Engagement Rate ~5-6% | Acima da média do mercado para conteúdo tech |
| Conversion Rate ~20% | Base de seguidores qualificada e engajada |
| From Home domina (~70%) | Audiência fiel — seguidores consomem o conteúdo ativamente |
| 27 Captions reciclados | Conteúdo evergreen performa bem mesmo republicado |
| Posts virais (>10k impressões) ~10% do total | Alta concentração de resultado em poucos posts |
| #python e #datascience são as hashtags âncora | Nicho de conteúdo bem definido e consistente |

---

## 📁 Arquivos Gerados

| Arquivo | Conteúdo |
|---|---|
| `Instagram_data_tratado.csv` | Dataset limpo com todas as features criadas |
| `Instagram_data_modelagem.csv` | Dataset normalizado pronto para Machine Learning |
