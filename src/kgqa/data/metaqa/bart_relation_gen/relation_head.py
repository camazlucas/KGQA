import json
import math

import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartScaledWordEmbedding

# As 9 relacoes originais do KB do MetaQA (methods/DoG/KBQA_TASK/metaqa/dataset/kb.txt).
# Os caminhos gold do MetaQA usam a versao "_inv" de cada uma para travessia reversa.
FORWARD_RELATIONS = [
    "directed_by",
    "has_genre",
    "has_imdb_rating",
    "has_imdb_votes",
    "has_tags",
    "in_language",
    "release_year",
    "starred_actors",
    "written_by",
]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
PAD_ID, BOS_ID, EOS_ID = range(len(SPECIAL_TOKENS))


def build_relation_vocab():
    """Vocabulario fechado do decoder: PAD/BOS/EOS + as 9 relacoes + suas 9 versoes _inv."""
    relations = FORWARD_RELATIONS + [f"{r}_inv" for r in FORWARD_RELATIONS]
    return SPECIAL_TOKENS + relations


def relation_display_text(relation):
    """Texto usado so para inicializar o embedding da relacao a partir do BART pre-treinado."""
    if relation.endswith("_inv"):
        return "reverse of " + relation[: -len("_inv")].replace("_", " ")
    return relation.replace("_", " ")


def attach_relation_head(model, tokenizer, vocab):
    """
    Substitui o embedding do decoder e a lm_head do BART (que projetam para o
    vocabulario inteiro, ~50 mil sub-palavras) por uma cabeca pequena restrita
    ao vocabulario fechado de relacoes do MetaQA. O encoder mantem intactos o
    tokenizer/embeddings originais do BART, entao a pergunta continua sendo
    lida em linguagem natural -- so o lado do decoder fica restrito.

    Modifica `model` in-place e tambem o retorna, por conveniencia.
    """
    d_model = model.config.d_model
    embed_scale = math.sqrt(d_model) if model.config.scale_embedding else 1.0
    base_embeddings = model.get_input_embeddings().weight.data

    new_embed = BartScaledWordEmbedding(len(vocab), d_model, PAD_ID, embed_scale=embed_scale)
    new_lm_head = nn.Linear(d_model, len(vocab), bias=False)

    with torch.no_grad():
        new_embed.weight[PAD_ID] = base_embeddings[tokenizer.pad_token_id]
        new_embed.weight[BOS_ID] = base_embeddings[tokenizer.bos_token_id]
        new_embed.weight[EOS_ID] = base_embeddings[tokenizer.eos_token_id]

        for i, relation in enumerate(vocab[len(SPECIAL_TOKENS):], start=len(SPECIAL_TOKENS)):
            text = relation_display_text(relation)
            subword_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            new_embed.weight[i] = base_embeddings[subword_ids].mean(dim=0)

        # Mesmos valores iniciais do embedding de entrada, mas como parametro
        # independente (nao o mesmo tensor): o HF Trainer detecta pesos com o
        # mesmo tensor por identidade e os remove do checkpoint salvo, esperando
        # reconstruir o par pelo mapeamento padrao do BART (lm_head -> embedding
        # do encoder), que nao conhece essa cabeca customizada.
        new_lm_head.weight.copy_(new_embed.weight)

    model.model.decoder.embed_tokens = new_embed
    model.lm_head = new_lm_head
    del model.final_logits_bias
    model.register_buffer("final_logits_bias", torch.zeros(1, len(vocab)))

    # Config e generation_config sao independentes em transformers>=5: generate()
    # le do generation_config, forward()/shift_tokens_right leem do config.
    model.config.tie_word_embeddings = False
    model.config.vocab_size = len(vocab)  # forward() usa isso pra dar reshape na loss
    for cfg in (model.config, model.generation_config):
        cfg.pad_token_id = PAD_ID
        cfg.bos_token_id = BOS_ID
        cfg.eos_token_id = EOS_ID
        cfg.decoder_start_token_id = BOS_ID

    # O generation_config do bart-base traz parametros pensados pro vocabulario
    # original que, com a cabeca de 21 tokens, passam a apontar pra tokens errados:
    # forced_bos_token_id=0 forcaria o 1o token gerado a ser <pad> (id 0 aqui),
    # o que zerava o exact_match. Tambem removemos forced_eos, no_repeat_ngram_size
    # e o beam search default (num_beams=4); quem quiser beam passa num_beams
    # explicitamente no generate().
    gen_cfg = model.generation_config
    gen_cfg.forced_bos_token_id = None
    gen_cfg.forced_eos_token_id = None
    gen_cfg.no_repeat_ngram_size = None
    gen_cfg.num_beams = 1
    gen_cfg.early_stopping = False

    return model


def save_relation_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_relation_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
