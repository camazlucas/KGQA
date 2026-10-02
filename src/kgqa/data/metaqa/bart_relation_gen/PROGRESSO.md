# BART relation generation (MetaQA) — Progresso

Branch `experiment/bart-relation-finetune`, criada a partir da `experiment/rel-paths-datasets`
(que já tinha os CSVs do MetaQA e a avaliação do BeamQA). Objetivo: treinar um gerador próprio de
cadeia de relações (substituindo o módulo 1 do BeamQA, que previa a cadeia a partir só da pergunta
com a topic entity mascarada como `"ne"`), usando os CSVs gold já construídos, e comparar contra o
resultado de 100% Hits@1/Hits@3 do BeamQA original — que é enganoso, porque o MetaQA gera perguntas
por template fixo por relação (ver `evaluation/beamqa/PROGRESSO.md`).

## Decisões de desenho

- **Input**: a pergunta com a **entidade real** (não mascarada) — já vem assim na coluna `question`
  dos CSVs (`build_relpaths_dataset.py` substitui `[entidade]` por texto puro).
- **Escopo**: um único modelo treinado no split `allhops` (os 3 níveis de hop juntos), em vez de um
  modelo por hop.
- **Checkpoint base**: `facebook/bart-base` (não `bart-large`) — domínio fechado e pequeno (só
  filmes), alinhado ao objetivo de baixo custo computacional da dissertação.
- **Cabeça do decoder (a diferença central em relação a um fine-tuning padrão)**: em vez de manter a
  `lm_head` original do BART (projeta pro vocabulário inteiro, ~50 mil sub-palavras), o embedding do
  decoder e a `lm_head` são substituídos por uma cabeça pequena restrita ao **vocabulário fechado de
  relações do MetaQA**: as 9 relações de `methods/DoG/KBQA_TASK/metaqa/dataset/kb.txt`
  (`directed_by`, `has_genre`, `has_imdb_rating`, `has_imdb_votes`, `has_tags`, `in_language`,
  `release_year`, `starred_actors`, `written_by`) mais as 9 versões `_inv` (travessia reversa, usada
  nos caminhos gold) mais PAD/BOS/EOS — 21 tokens no total. O **encoder mantém** o vocabulário e os
  embeddings originais do BART (a pergunta continua sendo lida em linguagem natural). Implementado em
  `relation_head.py` (`attach_relation_head`).
  - Cada hop da cadeia gold vira **1 token de saída** (não mais texto com `"|"` como separador) — ex.:
    um exemplo de 2 hops tem `labels = [rel_a_id, rel_b_id, EOS_ID]`.
  - As 5 relações que nunca aparecem invertidas nos dados atuais (`has_genre`, `has_imdb_rating`,
    `has_imdb_votes`, `in_language`, `release_year` — são atributos, não ligam duas entidades) ainda
    entram no vocabulário por completude do esquema do KB, mesmo sem exemplos de treino para a
    direção inversa.
  - **Inicialização dos embeddings novos**: em vez de aleatória, cada token de relação é inicializado
    com a média dos embeddings pré-treinados das sub-palavras do nome da relação (com `_` trocado por
    espaço, ex. `"directed_by"` → `"directed by"`; para `_inv`, `"reverse of directed by"|`),
    tokenizado com o tokenizer original do BART. PAD/BOS/EOS pegam a embedding pré-treinada dos
    tokens pad/bos/eos originais.
  - O peso de saída (`lm_head`) começa com os mesmos valores do embedding de entrada, mas como
    **parâmetro independente** (não o mesmo tensor) — se fossem o mesmo tensor, o mecanismo de
    "tied weights" do HF Transformers (que detecta pesos compartilhados pela identidade do tensor)
    removeria um dos dois do checkpoint salvo durante o treino, esperando reconstruí-lo a partir do
    mapeamento padrão do BART (`lm_head` ↔ embedding do encoder), que não sabe dessa cabeça
    customizada.
  - **Cuidado ao persistir**: `model.save_pretrained()`/`from_pretrained()` não são usados para o
    modelo treinado, porque salvariam `config.vocab_size` já reduzido pro vocabulário pequeno do
    decoder (necessário só porque o `forward()` do BART usa esse campo pra dar `reshape` na loss) —
    reconstruir com `from_pretrained` a partir desse config quebraria o encoder, que precisa do
    vocabulário grande original. Em vez disso: `torch.save(model.state_dict(), ...)` +
    `relation_vocab.json` (a lista de relações, na ordem usada para os ids) + tokenizer salvo à
    parte. Quem for carregar o modelo depois (`predict.py`) precisa reconstruir a arquitetura
    chamando `attach_relation_head` de novo antes de carregar os pesos.
  - **`generation_config` herdado do `bart-base`**: `attach_relation_head` precisa limpar os parâmetros
    de geração do checkpoint original, que apontam para ids do vocabulário grande. Com
    `forced_bos_token_id=0` o `generate()` forçava o primeiro token gerado a ser o id 0, que na cabeça
    de 21 tokens é o `<pad>`, e o `exact_match` no `valid` ficava em 0 mesmo com a loss perto de zero
    (o greedy manual, sem `generate()`, acertava). Por isso `forced_bos_token_id`,
    `forced_eos_token_id` e `no_repeat_ngram_size` são zerados e `num_beams=1`/`early_stopping=False`
    viram o default; quem quiser beam search passa `num_beams` explicitamente em `generate()`. O
    `generation_config` não entra no `state_dict`, então modelos já treinados não precisam ser
    retreinados.

## Treino

`train.py`: lê `outputs/metaqa_allhops_{train,valid}.csv`, treina com `Seq2SeqTrainer`
(`facebook/bart-base` + cabeça customizada), avaliando por época no split `valid` via exact-match da
cadeia gerada (greedy). Usa `EarlyStoppingCallback` (paciência configurável, default 2) em vez de um
número fixo de épocas — com vocabulário de saída de só 21 classes e ~329 mil exemplos de treino, a
convergência deve ser rápida, e treinar além do ponto de estabilização no `valid` seria desperdício de
custo computacional sem ganho.

**Smoke test** (1 época, 500 exemplos, `--per_device_train_batch_size 8`) rodou de ponta a ponta sem
erros nesta máquina (CPU) — confirmou que forward/loss/geração/avaliação/salvamento funcionam com a
cabeça customizada. Nesse teste, `torch.cuda.is_available()` retornou `False`: **esta máquina não tem
GPU**. Extrapolando a vazão medida (~7 exemplos/seg no treino), a época completa do `allhops train`
(329.282 exemplos) levaria **~13h só de treino** — decisão: **o treino completo roda no PC do
laboratório (que tem GPU)**, não aqui. Fluxo: commitar/push daqui, `pull` no PC do laboratório (regra
do projeto: só `pull` lá, nunca `push`) e rodar `train.py` de lá.

Duas features adicionadas por causa da duração do treino completo (mesmo mais rápido em GPU, ainda é
um treino longo e sem supervisão constante):
- **Tempo de treino registrado**: `training_time.json` (`training_seconds`, `training_hours`,
  `epochs_trained`) salvo ao final, além de impresso no console.
- **Checkpoints periódicos de segurança**: a cada N épocas (`--checkpoint_every_n_epochs`, default 5),
  `PeriodicCheckpointCallback` salva uma cópia do modelo em `<output_dir>/epoch_<N>/` (mesmo esquema
  seguro de `torch.save` do checkpoint final, não `save_pretrained` — ver nota acima), independente do
  `save_strategy` do Trainer (que já salva/avalia a cada época pro early stopping, mas só mantém as 2
  últimas via `save_total_limit`). Serve de ponto de restauração caso o treino seja interrompido no
  meio.

Uso:
```
python -m src.kgqa.data.metaqa.bart_relation_gen.train
```

Saída (em `raw_data/bart_relation_gen/checkpoints/allhops/`, fora do git — pesos de modelo):
`model_state_dict.pt`, `relation_vocab.json`, tokenizer salvo, `training_args.json`,
`training_time.json`, mais os checkpoints periódicos `epoch_<N>/`.

## Próximas etapas

- `predict.py`: rodar inferência no split `test`, gerando as predições (greedy e/ou beam search para
  Hits@3) em formato comparável ao gold.
- `evaluate_bart_relation_gen.py`: comparar contra o gold por `qid` (mais simples que o casamento por
  pergunta mascarada usado em `evaluate_beamqa_relations.py`, já que aqui não há mascaramento) e contra
  os resultados do BeamQA original, com a ressalva de que o 100% do BeamQA pode ser só memorização de
  template.
