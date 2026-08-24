# Pré-processamento do GrailQA

## Objetivo

Esta etapa tem como objetivo preparar o dataset GrailQA para os experimentos de KGQA com LLMs.

O processamento compreende:

1. Extração dos gold paths;
2. Validação dos caminhos no Freebase;
3. Recuperação das triplas correspondentes aos caminhos;
4. Validação das entidades resposta;
5. Conversão dos Freebase MIDs para labels textuais.

O resultado final é um dataset textual utilizado nos experimentos com LLMs.

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

## Execução

O processamento deve ser executado a partir da raiz do projeto `KGQA`.

### Estrutura esperada

KGQA/
├── src/
│   └── kgqa/
│       └── data/
│           └── grailqa/
│               ├── grailqa_gold_paths_validated.json
│               ├── grailqa_triples_text.json
│               └── preprocess_triples.py
│
└── [armazenamento externo]
    └── mid2label.pkl

O arquivo `mid2label.pkl` é mantido fora do repositório devido ao seu tamanho e deve estar disponível no ambiente de execução.

Os scripts utilizam `argparse` para receber os caminhos dos arquivos e não dependem de caminhos absolutos específicos de uma máquina.

### Comando

A partir da raiz do projeto:

    python -m src.kgqa.data.grailqa.preprocess_triples --input src/kgqa/data/grailqa/grailqa_gold_paths_validated.json --dictionary <CAMINHO_PARA_mid2label.pkl> --output src/kgqa/data/grailqa/grailqa_triples_text.json

### Parâmetros

| Argumento | Descrição |
|---|---|
| `--input` | Dataset GrailQA validado |
| `--dictionary` | Caminho para o dicionário MID → label |
| `--output` | Caminho do dataset textual gerado |

Os caminhos podem ser adaptados ao ambiente de execução.