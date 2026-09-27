# Avaliação do gerador de relações do BeamQA (MetaQA) — Progresso

Avalia a precisão do componente de geração de cadeia de relações do **BeamQA original**
(pré-treinado, não o fork adaptado deste projeto) contra o gold do MetaQA, split de test. Essa
pasta (`evaluation/beamqa/`) é a primeira de uma série — cada modelo avaliado ganha sua própria
pasta em `evaluation/<modelo>/`.

## Fonte das predições

`methods/_original_reference/BeamQA/Data/Path_gen/outputs/`:
`predictions_metaqa_{1,2,3}hop_wscores.txt` (e também `wqsp_predictions_wscores.txt`, não avaliado
ainda — foco atual é só MetaQA).

Confirmado por contagem de linhas que são predições do split de **test** (cada arquivo tem ~1
exemplo a menos que o total de test, provavelmente truncamento no fim da geração — ver detalhe de
cobertura abaixo).

Formato por linha: índice, pergunta com a topic entity mascarada como `"ne"`, top-3 cadeias de
relação candidatas (separadas por `|`; relações dentro de uma cadeia separadas por espaço), scores
das 3 candidatas, e scores por hop de cada candidata.

## Metodologia

O BeamQA mascara a topic entity da pergunta (`"ne"`) antes de prever a cadeia de relação — ou seja,
a predição não depende de qual entidade é, só do texto/template da pergunta. Por isso, o script
(`evaluate_beamqa_relations.py`) casa cada exemplo gold com a predição **mascarando a topic entity
do gold da mesma forma** (substitui o valor de `topic_entities` por `"ne"` na `question`), e não por
posição/índice — confirmado empiricamente que a ordem das linhas no arquivo de predições **não**
corresponde à ordem do csv gold.

Métricas: **Hits@1** (a candidata #1 bate exatamente com a cadeia gold, na ordem certa) e
**Hits@3** (alguma das 3 candidatas bate). Match é da cadeia inteira, não parcial.

Uso:
```
python -m src.kgqa.data.metaqa.evaluation.evaluate_beamqa_relations
```

## Resultado

| hop | exemplos gold (test) | sem predição encontrada | avaliados | Hits@1 | Hits@3 |
|---|---|---|---|---|---|
| 1-hop | 9.947 | 377 | 9.570 | 100,00% | 100,00% |
| 2-hop | 14.872 | 1.051 | 13.821 | 100,00% | 100,00% |
| 3-hop | 14.274 | 0 | 14.274 | 100,00% | 100,00% |

Relatório completo: `beamqa_relation_eval_report.json`.

## Achado importante: 100% não significa que o modelo generaliza

As perguntas do MetaQA são geradas por **template fixo por relação** — confirmado no artigo
original do dataset, Zhang et al., *"Variational Reasoning for Question Answering with Knowledge
Graph"* (AAAI 2018, [arXiv:1709.04071](https://arxiv.org/pdf/1709.04071)): a versão "Vanilla" do
MetaQA vem do WikiMovies com perguntas templatizadas por tipo de relação, com uma variante
paraphraseada via tradução neural (inglês→francês→inglês) pra dar variação linguística mantendo o
mesmo significado.

Isso foi confirmado empiricamente nos dados: ao mascarar a topic entity, o **mesmo template de
pergunta aparece centenas de vezes** no arquivo de predições, sempre com a mesma cadeia de relação
prevista e o mesmo score (ex: `"what does ne appear in"` aparece ~90 vezes, sempre prevendo
`starred_actors_inv` com score ~0,996). Ou seja, prever a relação a partir da pergunta no MetaQA é,
na prática, um lookup determinístico de template→relação, não uma tarefa que exige generalização
real. **O 100% de Hits@1/Hits@3 reflete essa característica do benchmark, não necessariamente que o
BeamQA é perfeito** — é uma limitação do MetaQA como benchmark pra esse componente específico, algo
a ter em mente ao comparar modelos futuros também avaliados nele.

## Cobertura: "sem predição encontrada"

377 (1-hop) e 1.051 (2-hop) exemplos gold não encontraram um template correspondente no arquivo de
predições — provavelmente porque o truncamento de ~1 linha no fim da geração cortou um template raro
por completo (e, como vários exemplos gold podem compartilhar o mesmo template, perder 1 linha de
predição pode custar vários exemplos gold sem match). O 3-hop não teve nenhum caso, sugerindo que a
linha truncada ali era duplicata de um template já presente em outra linha. Não investigado a fundo
ainda — os exemplos sem predição ficam de fora do cálculo de Hits@1/Hits@3 (não contam como erro).

## Próximas etapas

- Avaliar os outros modelos (GCR, DoG, etc.) da mesma forma, cada um em `evaluation/<modelo>/`.
- Eventualmente avaliar o WQSP (`wqsp_predictions_wscores.txt`) também, quando o gold de test do
  WQSP for construído nesta branch (ver `src/kgqa/data/metaqa/PROGRESSO.md`, próximas etapas).
