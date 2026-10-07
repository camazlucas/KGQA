import pickle
import re

MID_PATTERN = re.compile(r"^[a-z]\.[0-9a-z_]+$")


def looks_like_mid(value):
    """Detecta se um valor ja e um Freebase MID (ex: 'm.02mjmr'), em vez de um label."""
    return bool(MID_PATTERN.match(value))


def load_label_to_mid_dict(dictionary_path):
    """
    Carrega o dicionario mid2label.pkl (MID -> label) e inverte para
    label -> lista de MIDs, ja que varios MIDs podem compartilhar o mesmo label.
    """
    print(f"Carregando dicionario MID<->label de {dictionary_path}...")
    with open(dictionary_path, "rb") as f:
        mid2label = pickle.load(f)

    label2mids = {}
    for mid, label in mid2label.items():
        label2mids.setdefault(label, []).append(mid)

    print(f"Dicionario carregado: {len(mid2label)} MIDs, {len(label2mids)} labels distintos.")
    return label2mids


def resolve_labels_to_mids(labels, label2mids):
    """
    Resolve uma lista de labels (topic_entities ou answers do RoG-webqsp/RoG-cwq,
    que vem como texto e nao como MID) para MIDs do Freebase, usando o dicionario
    invertido. Valores que ja parecem ser MIDs sao mantidos como estao.

    Retorna:
      - all_mids: lista achatada de todos os MIDs candidatos (uniao de todos os labels)
      - label_to_mids: dict {label_original: [mids candidatos]}, so para labels resolvidos
    """
    label_to_mids = {}
    all_mids = []

    for label in labels:
        if looks_like_mid(label):
            candidates = [label]
        else:
            candidates = label2mids.get(label)

        if candidates:
            label_to_mids[label] = candidates
            all_mids.extend(candidates)

    return all_mids, label_to_mids
