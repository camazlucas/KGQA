# MetaQA — Progresso

Parte da branch `experiment/rel-paths-datasets`, cujo objetivo é construir datasets no formato
`{qid, question, topic_entities, answers, paths}` (pergunta, topic entity, respostas e caminho de
relações) a partir de fontes prontas — em vez de tentar recuperar/inferir esses caminhos, como se
tentou na branch `experiment/rog-subgraph-extraction`. MetaQA foi o primeiro dataset dessa frente,
por já vir com tudo resolvido por nome de entidade (sem o problema de ambiguidade de MID que o
Freebase tem).

## Contexto: por que MetaQA é mais simples que WebQSP/CWQ

O KB do MetaQA é pequeno e fechado (domínio de filmes), e usa **nome da entidade direto como
identificador** (`head|relation|tail` em texto puro) — não há MID nem outro esquema de ID, então
não existe a ambiguidade de resolução que travou o trabalho no Freebase (ver
`kgs/freebase/rog-subgraph/PROGRESSO.md`, achado do label `"Brazil"` com 1.167 MIDs candidatos).

## Fontes encontradas

- **Grafo completo**: `methods/DoG/KBQA_TASK/metaqa/dataset/kb.txt` — 134.741 triplas,
  `head|relation|tail`.
- **Perguntas com caminho**: `methods/_original_reference/BeamQA/Data/QA_data/MetaQA/`, arquivos
  `{train,valid,test}_{1,2,3}hop.txt`. Formato por linha (3 campos separados por tab): pergunta com
  a topic entity entre colchetes, respostas separadas por `|`, cadeia de relações separada por `|`.
  Todas as linhas têm exatamente 3 campos (sem o problema de linhas incompletas que apareceu no
  `test_wqsp.txt` do BeamQA para WebQSP).
- O submodule `methods/BeamQA` (fork ativo, não o `_original_reference`) **não tem** esses arquivos
  de QA do MetaQA — por isso a fonte usada é o `_original_reference`.
- O **DoG** tem os mesmos dados reempacotados em `methods/DoG/KBQA_TASK/metaqa/dataset/{1,2,3}-hop/`,
  com contagem de linhas idêntica à do BeamQA — confirma que é o mesmo benchmark canônico.

## Script

`build_relpaths_dataset.py` lê os arquivos de `{train,valid,test}_{1,2,3}hop.txt`, separa a topic
entity dos colchetes, e gera um csv por hop/split, mais um csv `allhops` por split (concatenando os
3 níveis de hop). Uso:

```
python -m src.kgqa.data.metaqa.build_relpaths_dataset
```

Schema por registro (csv, colunas `qid,question,topic_entities,answers,paths`;
`topic_entities`/`answers`/`paths` são listas juntadas por `|` dentro da célula, mesma convenção do
arquivo original):

```
qid,question,topic_entities,answers,paths
metaqa_2hop_test_0000,"which person directed the movies starred by John Krasinski","John Krasinski","Nancy Meyers|Sam Mendes|George Clooney|Ken Kwapis|Luke Greenfield","starred_actors_inv|directed_by"
```

## Decisão: CSV em vez de JSON

A primeira versão salvou em JSON com indentação, e o maior arquivo (`allhops_train`) ficou com
**147MB**. Comparação de tamanho pro caso `1hop_train` (96.106 exemplos):

| formato | tamanho |
|---|---|
| JSON com indent | 27,10 MB |
| JSON compacto (sem indent) | 18,22 MB |
| CSV | 11,42 MB |
| `.txt` original (BeamQA) | 7,85 MB |

JSON é mais caro porque repete o nome de cada campo em todo registro; CSV só escreve o cabeçalho
uma vez. Optou-se por CSV — mais compacto, ao custo de `topic_entities`/`answers`/`paths` serem
strings com `|` em vez de listas nativas (quem consumir precisa fazer `.split("|")`).

## Resultado final (12 arquivos, `outputs/`)

| arquivo | exemplos | tamanho |
|---|---|---|
| `metaqa_1hop_train.csv` | 96.106 | 11,4 MB |
| `metaqa_1hop_valid.csv` | 9.992 | 1,2 MB |
| `metaqa_1hop_test.csv` | 9.947 | 1,2 MB |
| `metaqa_2hop_train.csv` | 118.980 | 25,2 MB |
| `metaqa_2hop_valid.csv` | 14.872 | 3,2 MB |
| `metaqa_2hop_test.csv` | 14.872 | 3,1 MB |
| `metaqa_3hop_train.csv` | 114.196 | 35,6 MB |
| `metaqa_3hop_valid.csv` | 14.274 | 4,4 MB |
| `metaqa_3hop_test.csv` | 14.274 | 4,5 MB |
| `metaqa_allhops_train.csv` | 329.282 | 72,3 MB |
| `metaqa_allhops_valid.csv` | 39.138 | 8,8 MB |
| `metaqa_allhops_test.csv` | 39.093 | 8,7 MB |

`metaqa_allhops_train.csv` (72,3MB) está acima dos 50MB que o GitHub avisa (recomenda Git LFS), mas
abaixo do limite de 100MB que bloqueia o push — decisão foi versionar mesmo assim.

## Próximas etapas

- WebQSP via BeamQA (`test_wqsp.txt`): mesmo formato, mas com 34 linhas incompletas (23 sem
  relação, 11 sem caminho nenhum) a tratar; só o split de test do BeamQA tem caminho pronto, splits
  de train/validation ainda precisam de outra fonte ou processo.
- GrailQA: fonte ainda não investigada.
- Outros datasets, a definir.
