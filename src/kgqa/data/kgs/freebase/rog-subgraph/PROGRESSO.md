# RoG Subgraph — Progresso

Documenta o histórico do trabalho nesta pasta (`kgs/freebase/rog-subgraph/`), que lida com o
subgrafo do Freebase pré-extraído pelo RoG para o WebQSP e o CWQ (`rmanluo/RoG-webqsp`,
`rmanluo/RoG-cwq`) — não com processamento de QA em si.

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

## Próxima etapa

O objetivo final não é recuperar as respostas em si, e sim os **caminhos** (topic entity → …
→ answer). Com os MIDs de topic entity e answer garantidos (via `WebQSP.json`/`cwq.json` do DoG,
para o split de test), o plano é:

1. Montar, para `webqsp/test` e `cwq/test`, os pares de MID (topic entity, answer) usando o
   dataset do DoG em vez da resolução label→MID via dicionário.
2. Buscar o caminho **mais curto** entre os dois MIDs no Virtuoso, com **BFS bidirecional**
   (expandindo dos dois lados ao mesmo tempo até se encontrarem) — mais eficiente e mais
   confiável que o BFS de fonte única de `find_missing_answer_paths.py`, já que agora se sabe o
   destino exato de antemão.
3. Train e validation continuam em aberto — o DoG não tem uma fonte equivalente pra esses
   splits, então a resolução label→MID via `mid2label.pkl` (com a alta taxa de ambiguidade já
   registrada acima) ainda é a única opção conhecida até agora, a menos que se encontre outra
   fonte.
4. Caso fique muito extenso, montar o subgrafo apenas com o conteúdo completo que esteja no dataset do RoG. Ou seja, descartar as perguntas sem resposta no subgrafo e trabalhar apenas com as que tem todas as respostas.
