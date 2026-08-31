# Pré-processamento do GrailQA

## Objetivo

Esta etapa tem como objetivo preparar o dataset GrailQA para os experimentos de KGQA com LLMs.

O processamento compreende:

1. Extração dos gold paths;
2. Validação dos caminhos no Freebase;
3. Recuperação das triplas correspondentes aos caminhos;
4. Validação das entidades resposta;
5. Reconstrução dos caminhos concretos a partir das triplas;
6. Conversão dos Freebase MIDs para labels textuais.

O resultado é um dataset que mantém as representações estrutural e textual das informações recuperadas, permitindo seu uso nos experimentos com LLMs.

---

## Dataset original

O conjunto de validação do GrailQA possui **1000 exemplos**.

Durante a preparação do dataset, foram descartados exemplos que não atendiam aos requisitos do experimento.

### Exemplos sem `topic_entity`

Foram descartados **12 exemplos** sem `topic_entity`.

### Funções incompatíveis

Também foram descartados **25 exemplos** cuja função não representa uma consulta baseada exclusivamente na formação de caminhos:

- `argmin`: 1
- `>=`: 1
- `argmax`: 15
- `<=`: 4
- `count`: 3
- `>`: 1

Após essas filtragens:

```text
1000 exemplos
- 12 sem topic_entity
- 25 com funções incompatíveis
= 963 exemplos válidos


### Freebase, validação e conversão MID → Label
```

---

## Validação dos Gold Paths no Freebase

Após a extração dos caminhos, os relation paths são executados no Freebase para recuperar as triplas correspondentes.

O Freebase utilizado é o dump completo, armazenado em `fb_en.txt`, com aproximadamente **51 GB descompactado**.

Devido ao tamanho do Knowledge Graph, as consultas são realizadas utilizando o **Virtuoso** através de **SPARQL**, em vez de carregar o KG completo em estruturas Python na memória.

Para cada relação do caminho, são recuperadas todas as triplas compatíveis com a entidade atual.

As triplas são armazenadas no formato:

```json
{
  "head": "m.0yrltsn",
  "relation": "theater.play.productions",
  "tail": "m.0yrlqjm"
}
```

As relações reversas são mantidas na representação do gold path utilizando a notação `(R relation)`.

## Validação das Answers

Após a recuperação das triplas, é verificado se cada entidade `answer` está presente nas triplas recuperadas.

Dos 963 exemplos válidos inicialmente:

```
963 exemplos
962 answers encontradas
1 answer não encontrada
```
O exemplo cuja answer não foi encontrada é descartado.

Assim, o dataset final utilizado nas etapas seguintes possui:

```
962 exemplos
962 answers encontradas
0 answer não encontrada
```

## Reconstrução dos Caminhos Concretos

Após a validação das triplas, os caminhos abstratos do GrailQA são combinados com as triplas recuperadas para reconstruir os caminhos concretos percorridos no Knowledge Graph.

Essa etapa é realizada pelo script:

`generate_concrete_paths.py`

O caminho concreto preserva as entidades como Freebase MIDs e as relações na ordem de navegação.

Por exemplo, uma tripla armazenada como:

```
m.0yrltsn
→ theater.play.productions
→ m.0yrlqjm
```

quando utilizada por um caminho reverso:

`(R theater.play.productions)`

é reconstruída como:

```
m.0yrlqjm
→ (R theater.play.productions)
→ m.0yrltsn
```

O resultado é armazenado em:

`grailqa_concrete_paths.json`

Esse dataset mantém as informações anteriores e adiciona o campo `concrete_paths`.

A estrutura contém:

```
qid
question
topic_entities
paths
answers
kg_results
concrete_paths
```

Assim, `kg_results` pode ser utilizado como representação em triplas e `concrete_paths` como representação em caminhos concretos.

## Conversão MID → Label

Após a reconstrução dos caminhos concretos, os Freebase MIDs são convertidos para labels textuais utilizando o dicionário:

`mid2label.pkl`

A conversão é realizada pelo script:

`preprocess_grailqa_labels.py`

São convertidos:

1. `head` das triplas;
2. `tail` das triplas;
3. `topic_entities`;
4. `answers`;
5. entidades presentes em `concrete_paths`.

As relações e os gold paths não são convertidos.

Caso um MID não esteja presente no dicionário, o próprio MID é mantido como fallback e contabilizado como `missing_labels`.

As colisões de labels são mantidas intencionalmente. Assim, entidades distintas do Freebase podem possuir a mesma representação textual.

O resultado final é:

`grailqa_paths_text.json`

## Execução

O processamento deve ser executado a partir da raiz do projeto `KGQA`.

### Estrutura esperada

KGQA/
├── src/
│   └── kgqa/
│       └── data/
│           └── grailqa/
│               ├── generate_concrete_paths.py
│               ├── preprocess_grailqa_labels.py
│               └── outputs/
│                   ├── grailqa_gold_paths_validated.json
│                   ├── grailqa_concrete_paths.json
│                   └── grailqa_paths_text.json
│
└── [armazenamento externo]
    └── mid2label.pkl

O arquivo `mid2label.pkl` é mantido fora do repositório devido ao seu tamanho e deve estar disponível no ambiente de execução.

Os scripts utilizam `argparse` para receber os caminhos dos arquivos e não dependem de caminhos absolutos específicos de uma máquina.

### Geração dos Caminhos Concretos

A partir da raiz do projeto:

    python -m src.kgqa.data.grailqa.generate_concrete_paths \
    --input src/kgqa/data/grailqa/outputs/grailqa_gold_paths_validated.json \
    --output src/kgqa/data/grailqa/outputs/grailqa_concrete_paths.json



### Conversão MID -> Label

A partir da raiz do projeto:

    python -m src.kgqa.data.grailqa.preprocess_grailqa_labels \
    --input src/kgqa/data/grailqa/outputs/grailqa_concrete_paths.json \
    --dictionary <CAMINHO_PARA_mid2label.pkl> \
    --output src/kgqa/data/grailqa/outputs/grailqa_paths_text.json

### Parâmetros

| Argumento | Descrição |
|---|---|
| `--input` | Dataset GrailQA validado |
| `--dictionary` | Caminho para o dicionário MID → label |
| `--output` | Caminho do dataset textual gerado |

Os caminhos podem ser adaptados ao ambiente de execução.