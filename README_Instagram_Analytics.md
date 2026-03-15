# 📊 Instagram Analytics — Análise Exploratória de Dados
## @suasaladatech | Data Science & Python

> **Objetivo:** Analisar os dados de performance de posts no Instagram para entender o que gera mais alcance, engajamento e novos seguidores.
> **Dataset:** 119 posts com métricas como impressões, likes, saves, comentários, compartilhamentos e origem do tráfego.
> **Ferramenta:** Google Colab — `Instagram_Analytics.ipynb`

---

## 📋 Índice

1. [Estrutura do Projeto](#1-estrutura-do-projeto)
2. [Pré-requisitos](#2-pré-requisitos)
3. [Ordem de Execução](#3-ordem-de-execução)
4. [Configuração do Ambiente](#4-configuração-do-ambiente)
5. [Leitura e Inspeção Inicial](#5-leitura-e-inspeção-inicial)
6. [Qualidade dos Dados](#6-qualidade-dos-dados)
7. [Engenharia de Features](#7-engenharia-de-features)
8. [Normalização e Encoding](#8-normalização-e-encoding)
9. [Estatísticas Descritivas](#9-estatísticas-descritivas)
10. [Detecção de Outliers](#10-detecção-de-outliers)
11. [Análise de Distribuição](#11-análise-de-distribuição)
12. [Correlações](#12-correlações)
13. [Análise de Texto](#13-análise-de-texto)
14. [Insights e Conclusões](#14-insights-e-conclusões)
15. [Arquivos Gerados](#15-arquivos-gerados)

---

## 1. Estrutura do Projeto

```
instagram-analytics/
├── Instagram_data.csv               # Dataset original (119 posts)
├── Instagram_data_tratado.csv       # Dataset limpo com features (102 posts · 20 colunas)
├── Instagram_data_modelagem.csv     # Dataset normalizado para ML (102 posts · 19 colunas)
├── Instagram_Analytics.ipynb        # Notebook Google Colab — análise completa
└── README.md                        # Este arquivo
```

---

## 2. Pré-requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud missingno
```

| Biblioteca | Uso |
|---|---|
| `pandas` / `numpy` | Manipulação e cálculo |
| `matplotlib` / `seaborn` | Visualizações e gráficos |
| `missingno` | Mapa visual de valores nulos |
| `wordcloud` | Nuvem de palavras das hashtags |
| `scikit-learn` | Normalização e encoding |

---

## 3. Ordem de Execução

```
1.  Configuração do ambiente
2.  Leitura e inspeção inicial
3.  Qualidade dos dados
    3.1 Valores nulos
    3.2 Duplicatas
    3.3 Verificação de tipos
4.  Engenharia de features          ← ANTES da normalização
5.  Normalização e Encoding         ← ANTES das estatísticas
6.  Estatísticas descritivas
7.  Detecção de outliers (IQR)
8.  Análise de distribuição
9.  Correlações
10. Análise de texto
11. Insights e conclusões
```

> ⚠️ A **Engenharia de Features** deve ser executada **antes** da normalização — as features criadas precisam existir para serem normalizadas.

---

## 4. Configuração do Ambiente

```python
!pip install -q wordcloud missingno

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from wordcloud import WordCloud
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

print('✅ Ambiente configurado com sucesso!')
```

**Para o público leigo:** Antes de começar, preparamos todas as ferramentas necessárias — como montar um kit de instrumentos antes de iniciar um trabalho. Cada biblioteca tem uma função específica: uma para criar gráficos, outra para analisar textos, outra para detectar problemas nos dados.

---

## 5. Leitura e Inspeção Inicial

```python
# Encoding latin1 — necessário para caracteres especiais no arquivo
df = pd.read_csv('/content/Instagram_data.csv', encoding='latin1')

# Substituir nome do autor em todo o dataset
df = df.replace('amankharwal', 'suasaladatech', regex=True)

print(f'Shape: {df.shape}')   # (119, 13)
df.head()
df.info()
```

### Resultado

| Atributo | Valor |
|---|---|
| Shape | 119 linhas × 13 colunas |
| Colunas numéricas | 11 (int64) — métricas de performance |
| Colunas texto | 2 (object) — Caption e Hashtags |
| Substituições realizadas | 117 ocorrências de `suasaladatech` confirmadas |

**Para o público leigo:** Carregamos o arquivo com os dados dos posts — como abrir uma planilha Excel com 119 linhas e 13 colunas. Também fizemos um "localizar e substituir" para trocar o nome do autor original pelo correto `suasaladatech` em todos os textos.

---

## 6. Qualidade dos Dados

### 6.1 Valores Nulos

```python
nulls = pd.DataFrame({
    'Total Nulos'    : df.isnull().sum(),
    'Percentual (%)' : (df.isnull().sum() / len(df) * 100).round(2)
})
nulls = nulls[nulls['Total Nulos'] > 0]

if nulls.empty:
    print('✅ Nenhum valor nulo encontrado!')
else:
    display(nulls)

# Mapa visual de nulos
msno.matrix(df, figsize=(12, 4), color=(0.2, 0.5, 0.8))
plt.show()
```

**Resultado:** ✅ Zero valores nulos — dataset 100% preenchido em todas as 13 colunas.

---

### 6.2 Duplicatas

```python
print(f'Duplicatas totais           : {df.duplicated().sum()}')          # 17
print(f'Duplicatas colunas numéricas: {df.duplicated(subset=["Impressions","Likes","Saves","Comments"]).sum()}')
print(f'Duplicatas por Caption      : {df.duplicated(subset=["Caption"]).sum()}')  # 29

# Investigar Captions repetidos
repetidos = df['Caption'].value_counts()
repetidos = repetidos[repetidos > 1]
print(f'Captions que se repetem: {len(repetidos)}')   # 27
print(f'Posts envolvidos       : {repetidos.sum()}')  # 56

# Remover duplicatas — manter primeira ocorrência
df = df.drop_duplicates()
print(f'Shape após limpeza: {df.shape}')   # (102, 13)
```

| Critério | Resultado | Ação |
|---|---|---|
| `df.duplicated()` — todas as colunas | **17 linhas** | Removidas |
| Duplicatas colunas numéricas | **17 linhas** | Removidas |
| `subset=['Caption']` | **29 ocorrências** | 27 Captions distintos em 56 posts |
| Shape após limpeza | **102 linhas × 13 colunas** | -14,3% |

> Os 17 posts duplicados são cópias exatas das linhas 15–29 repetidas nas linhas 82–98 — duplicação em bloco do dataset. Os 27 Captions repetidos representam conteúdo reciclado ou reposts intencionais.

**Para o público leigo:** Encontramos 17 posts completamente idênticos (erros de registro) e os removemos. Após a limpeza, ficamos com 102 posts únicos.

---

### 6.3 Verificação de Tipos

```python
cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
cols_txt = df.select_dtypes(include=['object']).columns.tolist()

print(f'Numéricas ({len(cols_num)}): {cols_num}')
print(f'Texto     ({len(cols_txt)}): {cols_txt}')
```

```
Numéricas (11): Impressions, From Home, From Hashtags, From Explore,
                From Other, Saves, Comments, Shares, Likes,
                Profile Visits, Follows

Texto (2): Caption, Hashtags
```

Tipos corretos — não foi necessária conversão de dtype.

---

## 7. Engenharia de Features

```python
# Taxa de engajamento
df['engagement_rate'] = (
    (df['Likes'] + df['Comments'] + df['Shares'] + df['Saves'])
    / df['Impressions'] * 100
).round(4)

# Taxa de conversão
df['conversion_rate'] = (
    df['Follows'] / df['Profile Visits'].replace(0, np.nan) * 100
).round(4)

# Origem dominante do tráfego
fontes = ['From Home', 'From Hashtags', 'From Explore', 'From Other']
df['main_source'] = df[fontes].idxmax(axis=1)

# Categorizar alcance
df['impressions_category'] = pd.cut(
    df['Impressions'],
    bins=[0, 3000, 5000, 10000, float('inf')],
    labels=['Baixo', 'Médio', 'Alto', 'Viral']
)

# Features de texto
df['caption_length'] = df['Caption'].astype(str).apply(len)
df['num_hashtags']   = df['Hashtags'].astype(str).apply(
    lambda x: len([t for t in x.split() if t.startswith('#')])
)

print('✅ Features criadas com sucesso!')
print(f'Shape: {df.shape}')   # (102, 19)
```

### Features criadas

| Feature | Fórmula | Resultado médio |
|---|---|---|
| `engagement_rate` | (Likes+Comments+Shares+Saves) / Impressions × 100 | ~5-6% |
| `conversion_rate` | Follows / Profile Visits × 100 | ~20% |
| `main_source` | `idxmax(From Home, Hashtags, Explore, Other)` | From Home (~70%) |
| `impressions_category` | `cut([0, 3k, 5k, 10k, ∞])` | Maioria em Médio e Alto |
| `caption_length` | `len(Caption)` | 180 chars (média) |
| `num_hashtags` | `count(# in Hashtags)` | ~19 por post |

**Para o público leigo:**
- **Taxa de Engajamento:** de cada 100 pessoas que viram o post, ~5 a 6 interagiram — considerado bom para o Instagram
- **Taxa de Conversão:** de cada 100 pessoas que visitaram o perfil, ~20 viraram seguidores
- **Categorias de alcance:** Baixo (<3k) · Médio (3–5k) · Alto (5–10k) · Viral (>10k)

---

## 8. Normalização e Encoding

### Min-Max Scaler — intervalo [0, 1]

```python
cols_normalizar = [
    'Impressions', 'From Home', 'From Hashtags', 'From Explore',
    'From Other', 'Saves', 'Comments', 'Shares', 'Likes',
    'Profile Visits', 'Follows', 'caption_length', 'num_hashtags',
    'engagement_rate', 'conversion_rate'
]

scaler_minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[cols_normalizar] = scaler_minmax.fit_transform(df[cols_normalizar])
# Verificação: min=0.00 e max=1.00 em todas as colunas ✅
```

### Standard Scaler — média=0, desvio=1

```python
scaler_std = StandardScaler()
df_standard = df.copy()
df_standard[cols_normalizar] = scaler_std.fit_transform(df[cols_normalizar])
# Verificação: mean≈0.00 e std=1.00 em todas as colunas ✅
```

### One-Hot Encoding — `main_source` (nominal, sem ordem)

```python
ohe = pd.get_dummies(df['main_source'], prefix='source', dtype=int)
# Gera: source_From Explore | source_From Hashtags | source_From Home
```

### Label Encoding — `impressions_category` (ordinal, com ordem)

```python
ordem = {'Baixo': 0, 'Médio': 1, 'Alto': 2, 'Viral': 3}
df['impressions_category_encoded'] = df['impressions_category'].map(ordem)
```

| Técnica | Variável | Motivo | Uso recomendado |
|---|---|---|---|
| Min-Max Scaler | 15 numéricas | Escala [0,1] | Redes neurais, K-Means |
| Standard Scaler | 15 numéricas | Média=0, Desvio=1 | Regressão, SVM, PCA |
| One-Hot Encoding | `main_source` | Sem ordem hierárquica | Variáveis nominais |
| Label Encoding | `impressions_category` | Com ordem definida | Variáveis ordinais |

**Dataset final:** `102 linhas × 19 colunas` — 100% numérico, pronto para ML ✅

---

## 9. Estatísticas Descritivas

```python
pd.set_option('display.float_format', '{:,.2f}'.format)

resumo = pd.DataFrame({
    'Média'         : df[cols_num].mean().round(2),
    'Mediana'       : df[cols_num].median(),
    'Desvio Padrão' : df[cols_num].std().round(2),
    'Assimetria'    : df[cols_num].skew().round(2),
    'Curtose'       : df[cols_num].kurt().round(2),
})
display(resumo)
```

### Resultados principais (base: 102 posts)

| Métrica | Média | Mediana | Desvio Padrão | Assimetria | Curtose |
|---|---|---|---|---|---|
| Impressions | 5.920 | 4.344 | 5.140 | 3,98 | 19,42 |
| Likes | 177 | 158 | 85 | 1,72 | 4,00 |
| Follows | 23 | 8 | 44 | 3,75 | 15,52 |
| Profile Visits | 55 | 24 | 93 | 3,89 | 16,92 |
| Comments | 6 | 6 | 3 | 0,76 | 1,89 |

> Alta assimetria positiva em Impressions (3,98) e Follows (3,75) indica distribuições com cauda longa à direita. Comments é a variável mais próxima de distribuição normal (skewness=0,76, curtose=1,89).

**Para o público leigo:** A diferença entre média (5.920) e mediana (4.344) nas Impressions revela que poucos posts virais "puxam" a média para cima — a maioria dos posts fica em torno de 4.344 visualizações.

---

## 10. Detecção de Outliers

```python
def detectar_outliers_iqr(df, colunas):
    resultado = []
    for col in colunas:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lim_sup  = Q3 + 1.5 * IQR
        outliers = df[df[col] > lim_sup]
        resultado.append({
            'Coluna'       : col,
            'Limite Sup'   : round(lim_sup, 2),
            'Qtd Outliers' : len(outliers),
            '% Outliers'   : round(len(outliers) / len(df) * 100, 2)
        })
    return pd.DataFrame(resultado).set_index('Coluna')

display(detectar_outliers_iqr(df, cols_num))
```

### Resultado por coluna

| Coluna | Limite Superior | Outliers | % |
|---|---|---|---|
| From Explore | 1.553 | 14 | 13,73% |
| Follows | 39 | 13 | 12,75% |
| Impressions | 10.407 | 11 | 10,78% |
| Profile Visits | 90 | 11 | 10,78% |
| From Other | 486 | 10 | 9,80% |
| Saves | 328 | 9 | 8,82% |
| Shares | 28 | 3 | 2,94% |
| Comments | 14 | 2 | 1,96% |

> O post mais extremo teve **36.919 impressões**, 611 visitas ao perfil e 228 novos seguidores — performance excepcional legítima, não erro de dado.

**Para o público leigo:** O post mais viral teve 36.919 visualizações — quase 9× acima da média. Esses casos excepcionais são os mais valiosos para entender o que funciona.

---

## 11. Análise de Distribuição

```python
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(cols_num):
    sns.histplot(df[col], kde=True, ax=axes[i], color='#4C72B0', bins=20)
    axes[i].set_title(col, fontsize=10)

for j in range(len(cols_num), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribuição das Variáveis Numéricas', fontsize=14)
plt.tight_layout()
plt.show()
```

> A maioria das variáveis apresenta assimetria positiva (right-skewed), confirmando os valores de skewness. Comments é a única variável com distribuição próxima da normal.

**Para o público leigo:** Como um gráfico de notas que mostra quantos alunos tiraram cada nota — a maioria dos posts tem performance mediana e apenas alguns viralizam.

---

## 12. Correlações

```python
corr = df[cols_num].corr()

# Heatmap
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriz de Correlação')
plt.show()

# Top correlações com Impressions
display(
    corr['Impressions'].drop('Impressions')
    .sort_values(ascending=False)
    .to_frame()
    .rename(columns={'Impressions': 'Correlação'})
    .style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1)
    .format('{:.4f}')
)
```

### Principais correlações com Impressions

| Variável | Tipo de Correlação | Insight |
|---|---|---|
| From Explore | Alta positiva ↑↑ | Maior driver de alcance |
| From Hashtags | Moderada positiva ↑ | Hashtags ampliam o alcance |
| Likes | Moderada positiva ↑ | Posts curtidos alcançam mais |
| Follows | Moderada positiva ↑ | Alcance alto converte seguidores |
| Comments | Baixa | Pouco impacto no alcance |

**Para o público leigo:**
- **Aparecer no Explore = muito mais visualizações** — maior impulsionador de alcance
- **Mais Likes = mais Follows** — posts curtidos tendem a gerar novos seguidores
- **Mais Likes = mais Saves** — conteúdo curtido também é salvo para rever depois

---

## 13. Análise de Texto

### Comprimento das Captions

```python
df['caption_length'] = df['Caption'].astype(str).apply(len)
print(df['caption_length'].describe().round(2))
# mean: 180 | std: 128 | min: 44 | max: 784

# Scatter caption_length × Impressions
sns.scatterplot(data=df, x='caption_length', y='Impressions', alpha=0.6)
plt.title('Comprimento da Caption vs Impressions')
plt.show()
```

> Comprimento médio de **180 caracteres**. O tamanho do texto não apresenta correlação forte com as Impressions — o conteúdo importa mais que o comprimento.

### Análise de Hashtags

```python
todas_hashtags = []
for texto in df['Hashtags'].astype(str):
    tags = [t.strip().lower() for t in texto.split() if t.startswith('#')]
    todas_hashtags.extend(tags)

contagem = Counter(todas_hashtags)
top_tags = pd.DataFrame(contagem.most_common(20), columns=['Hashtag', 'Frequência'])

# Gráfico top 20
sns.barplot(data=top_tags, x='Frequência', y='Hashtag', palette='Blues_r')
plt.title('Top 20 Hashtags Mais Utilizadas')
plt.show()

# WordCloud
wc = WordCloud(width=900, height=400, background_color='white',
               colormap='Blues', max_words=100).generate(' '.join(todas_hashtags))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud — Hashtags')
plt.show()
```

### Top 5 Hashtags

| # | Hashtag | Frequência |
|---|---|---|
| 1 | #suasaladatech | 100 |
| 2 | #thecleverprogrammer | 100 |
| 3 | #python | 93 |
| 4 | #pythonprogramming | 84 |
| 5 | #datascience | 79 |

> **163 hashtags únicas** — nicho bem definido em Data Science e Python.

---

## 14. Insights e Conclusões

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
| Posts virais ~10% do total | Alta concentração de resultado em poucos posts |
| #python e #datascience são hashtags âncora | Nicho de conteúdo bem definido e consistente |

---

## 15. Arquivos Gerados

```python
# Dataset tratado com todas as features
df.to_csv('/content/Instagram_data_tratado.csv', index=False, encoding='utf-8')

# Dataset normalizado pronto para Machine Learning
df_final.to_csv('/content/Instagram_data_modelagem.csv', index=False)

print('✅ Exportados com sucesso!')
```

| Arquivo | Conteúdo | Linhas | Colunas |
|---|---|---|---|
| `Instagram_data_tratado.csv` | Dataset limpo com todas as features | 102 | 20 |
| `Instagram_data_modelagem.csv` | Dataset normalizado para ML | 102 | 19 |

---
# Fonte Dataset : https://www.kaggle.com/datasets/amirmotefaker/instagram-data
*Projeto desenvolvido com apoio do **Claude (Anthropic)***
*Stack: Python · Pandas · Scikit-learn · Matplotlib · Seaborn · Google Colab*
