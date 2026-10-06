# RoG Subgraph — Progresso

Documenta o histórico do trabalho nesta pasta (`kgs/freebase/rog-subgraph/`), que lida com o
subgrafo do Freebase pré-extraído pelo RoG para o WebQSP e o CWQ (`rmanluo/RoG-webqsp`,
`rmanluo/RoG-cwq`) — não com processamento de QA em si. Estrutura: `scripts/` (os scripts
citados abaixo), `kg/` (subgrafo global mesclado, ver seção "Abordagem adotada"), `outputs/`
(relatórios e listas intermediárias em json).

## Contexto

`check_answer_coverage.py` verificou se as `a_entity` de cada exemplo aparecem no `graph`
pré-extraído do próprio exemplo. Achado: **8.329 exemplos (~21% de ~39.389)** têm pelo menos
uma resposta faltando no subgrafo pré-extraído — limitação conhecida do método de extração
(PPR com teto fixo de entidades, linhagem GraftNet/PullNet/NSM), não um bug. Resultado salvo em
`outputs/answer_coverage_report.json`. Breakdown por dataset/split (cobertura completa / total):

- WebQSP: train 2611/2826, validation 228/246, test 1501/1628
- CWQ: train 21242/27639, validation 2720/3519, test 2758/3531

## Etapas feitas

1. **`extract_missing_answers.py`** — a partir do relatório acima, monta por dataset/split um
   json `{qid, topic_entities, answers}` (`answers` = só as faltantes no subgrafo) para todo
   exemplo problemático. Rodado para os 6 combos (webqsp/cwq × train/validation/test), outputs em
   `outputs/missing_answers_<dataset>_<split>.json`.

2. **`find_missing_answer_paths.py`** — tentativa original de recuperar o caminho até as
   respostas faltantes: BFS multi-fonte a partir das `topic_entities`, direto no Freebase
   completo via SPARQL no Virtuoso local, até `max_hops` (default 3). Como `topic_entities` e
   `answers` do RoG vêm como **texto** (não MID), foi acoplada a resolução label→MID (via
   `mid_label.py`, em `kgs/freebase/`, usando o dicionário local `mid2label.pkl`). Testado em
   pequena escala (WebQSP train/validation/test, 10 exemplos cada) e **0 respostas recuperadas**
   nos 30 casos — sinal de que a resolução label→MID era o gargalo (ver achado abaixo). CWQ não
   chegou a ser testado nessa abordagem.

3. **`convert_labels_to_mids.py`** — isolou a etapa de resolução label→MID como um passo
   separado, pra investigar o problema acima antes de rodar o BFS em escala. Para cada label,
   classifica em: resolvido (1 MID candidato), ambíguo (2+ candidatos) ou sem MID (0
   candidatos). Um exemplo só entra no json principal
   (`outputs/missing_answers_mid_<dataset>_<split>.json`) se **tanto** `topic_entities` quanto
   `answers` sobrarem não-vazios após a resolução; os labels problemáticos ficam registrados em
   dois relatórios consolidados, `outputs/ambiguous_labels.json` (com a lista de MIDs candidatos)
   e `outputs/unresolved_labels.json`.

   **Achado**: o `mid2label.pkl` tem muito ruído em labels comuns — o label `"Brazil"`, por
   exemplo, resolve para **1.167 MIDs candidatos** (o MID real do país, `m.015fr`, está lá no
   meio). Isso derruba a taxa de sobrevivência de exemplos de forma severa:

   | split | exemplos sobreviventes | taxa |
   |---|---|---|
   | webqsp/train | 11/215 | 5% |
   | webqsp/validation | 0/18 | **0%** |
   | webqsp/test | 9/127 | 7% |
   | cwq/train | 1965/6397 | 31% |
   | cwq/validation | 229/799 | 29% |
   | cwq/test | 230/773 | 30% |

   Além disso, `ambiguous_labels.json` ficou com **77MB** (cada registro ambíguo guarda a lista
   completa de candidatos), o que estoura a convenção do projeto de manter arquivos grandes fora
   do versionamento. **Essa etapa ainda não foi commitada** — os outputs de
   `convert_labels_to_mids.py` (json principal por split, `ambiguous_labels.json`,
   `unresolved_labels.json`) estão no working tree, sem commit, enquanto se decide o que fazer
   com o tamanho do arquivo e com a taxa de sobrevivência baixa.

## Achado: dataset original do DoG resolve a ambiguidade (mas só para o split de test)

Dentro do submodule `methods/DoG/KBQA_TASK/freebase/dataset/` há 3 arquivos com os dados
*originais* (pré-RoG) de WebQSP/CWQ:

| arquivo | registros | cobertura | conteúdo |
|---|---|---|---|
| `WebQuestions.json` | 2.032 | só test | `topic_entity` com MID correto; `answers` só como label |
| `WebQSP.json` | 1.639 | só test | release oficial do WebQSP: `TopicEntityMid`, e cada resposta já vem com `AnswerArgument` (MID) + `EntityName` dentro de `Parses[].Answers[]`, além do **SPARQL original** que gerou a resposta |
| `cwq.json` | 3.531 | só test | `topic_entity` com MID, **SPARQL original**, `answer` como label (sem MID) |

Os ids batem **100%** com o RoG por join direto de qid: `WebQSP.json` cobre os 1.628/1.628
exemplos do `RoG-webqsp/test`, e `cwq.json` cobre os 3.531/3.531 do `RoG-cwq/test`. Nenhum dos
três cobre train/validation.

Isso elimina o problema de ambiguidade do `mid2label.pkl` para o split de test, sem precisar do
dicionário: `WebQSP.json` já dá o MID de topic entity e de answer diretamente; `cwq.json` dá o
MID da topic entity direto e, para a answer, dá pra rodar o SPARQL original do próprio exemplo
contra o Virtuoso local e pegar o MID certo (em vez de resolver por label).

## Plano anterior (superado pela abordagem abaixo)

O plano original era recuperar os **caminhos** (topic entity → … → answer) das respostas
faltantes via BFS bidirecional no Virtuoso, usando os MIDs sem ambiguidade do DoG (disponíveis
só para o split de test) e, para train/validation, a resolução label→MID via `mid2label.pkl`
(com a alta taxa de ambiguidade registrada acima). Esse plano foi abandonado em favor da
abordagem mais simples da seção seguinte, por decisão explícita de simplificar o escopo.

## Abordagem adotada: subgrafo global mesclado

Em vez de tentar recuperar as respostas faltantes, a estratégia virou: juntar todos os
subgrafos dos exemplos onde **pelo menos uma resposta já está presente** no subgrafo
pré-extraído, formando um único grafo global.

- **`build_merged_subgraph.py`** — para cada exemplo de WebQSP+CWQ (todos os splits), inclui o
  `graph` do exemplo no grafo global se pelo menos uma `a_entity` aparecer entre os nós do
  próprio subgrafo (ou seja, `fully_covered` + `partially_covered` na classificação do
  `check_answer_coverage.py`; exclui `zero_covered` e `no_answers`). Deduplica triplas exatas
  via `set()` do Python. Salva `kg/merged_subgraph.tsv` (`head\trelation\ttail`, uma tripla
  por linha) e `kg/merged_subgraph_stats.json` (contagem e qids inclusos por split).

- **`extract_entities_relations.py`** (movido para `src/kgqa/data/kgs/common/`, por ser
  genérico entre KGs; uso: `python -m src.kgqa.data.kgs.common.extract_entities_relations
  --graph src/kgqa/data/kgs/freebase/rog-subgraph/kg/merged_subgraph.tsv`) — a partir de
  `kg/merged_subgraph.tsv`, extrai as
  entidades e relações distintas (colunas head/tail e relation) e salva `kg/entities.txt`
  (2.493.661 entidades, ~43,5MB — fora do versionamento por tamanho) e `kg/relations.txt`
  (6.994 relações, ~300KB — versionado), uma por linha, ordenadas.

- **Decisão: mesclar por label de entidade, não por MID.** As entidades do campo `graph` do RoG
  são labels em texto (às vezes um MID bruto quando não há nome resolvido), não MIDs. Mesclar
  por label faz com que entidades reais diferentes com o mesmo nome (ex: duas coisas chamadas
  "Washington") virem o mesmo nó no grafo global — risco aceito conscientemente em troca de
  simplicidade. As relações já são predicados reais do Freebase (ex:
  `travel.tour_operator.travel_destinations`), então não têm esse problema de ambiguidade.

- **Resultado (rodado em escala completa, WebQSP + CWQ, todos os splits):**

  | dataset/split | exemplos qualificados | total |
  |---|---|---|
  | webqsp/train | 2715 | 2826 |
  | webqsp/validation | 235 | 246 |
  | webqsp/test | 1557 | 1628 |
  | cwq/train | 22089 | 27639 |
  | cwq/validation | 2848 | 3519 |
  | cwq/test | 2848 | 3531 |

  Total: **7.989.528 triplas únicas**, arquivo `kg/merged_subgraph.tsv` com ~525MB (fora do
  versionamento — entrada adicionada ao `.gitignore`).

- **Decisão: não carregar no Virtuoso nem converter para RDF/IRI.** Com ~525MB, o grafo cabe
  tranquilamente em RAM, então a ideia é trabalhá-lo direto em memória em Python (ex: dict de
  adjacência), sem precisar resolver entidades para MID nem codificar labels como IRI. Esse
  grafo mesclado é um conjunto **separado** do Freebase completo carregado no Virtuoso — sem
  cruzamento/join entre os dois.

## Próxima etapa

1. Carregar `kg/merged_subgraph.tsv` em uma estrutura de grafo em memória (ex: dict de
   adjacência direto/reverso, ou `networkx`), para uso em Python sem depender do Virtuoso.
2. Definir o que fazer com esse grafo em memória (ex: pathfinding entre topic entity e resposta
   via BFS bidirecional, direto no grafo mesclado) — ainda em aberto.
