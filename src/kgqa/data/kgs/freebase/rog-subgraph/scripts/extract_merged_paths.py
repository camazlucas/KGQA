import argparse
import csv
import json
import os

from datasets import load_dataset

DATASET_NAME_MAP = {
    "webqsp": "rmanluo/RoG-webqsp",
    "cwq": "rmanluo/RoG-cwq",
}
SPLIT_CHOICES = ["train", "validation", "test"]


def load_graph(graph_path):
    """
    Carrega o subgrafo mesclado (kg/merged_subgraph.tsv) numa adjacencia enxuta com
    entidades/relacoes convertidas para inteiros (economiza memoria vs. guardar string
    repetida em cada tripla).

    Para cada tripla (h, r, t), permite andar h->t (codigo par, relacao direta) e t->h
    (codigo impar, relacao inversa) -- mesma convencao de 'relacao inversa' usada no
    path_extraction.py de referencia (la com o rotulo "(R <rel>)").
    """
    entity_to_id = {}
    relation_list = []
    relation_to_id = {}
    adj = {}

    def entity_id(name):
        eid = entity_to_id.get(name)
        if eid is None:
            eid = len(entity_to_id)
            entity_to_id[name] = eid
            adj[eid] = []
        return eid

    def relation_id(name):
        rid = relation_to_id.get(name)
        if rid is None:
            rid = len(relation_list)
            relation_to_id[name] = rid
            relation_list.append(name)
        return rid

    with open(graph_path, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            h_id = entity_id(h)
            t_id = entity_id(t)
            r_id = relation_id(r)
            adj[h_id].append((r_id * 2, t_id))
            adj[t_id].append((r_id * 2 + 1, h_id))

    return entity_to_id, relation_list, adj


def relation_label(code, relation_list):
    r_id, is_inverse = divmod(code, 2)
    name = relation_list[r_id]
    return f"(R {name})" if is_inverse else name


def bidirectional_shortest_path(adj, source, target, max_hops):
    """
    BFS bidirecional (encontra do meio pros dois lados) entre dois ids de entidade.
    Retorna a lista de codigos de relacao do caminho mais curto (ou None se nao houver
    caminho dentro de max_hops).
    """
    if source == target:
        return None

    parent_fwd = {source: None}
    parent_bwd = {target: None}
    frontier_fwd = {source}
    frontier_bwd = {target}
    meeting_node = None

    for _ in range(max_hops):
        if not frontier_fwd or not frontier_bwd:
            break

        if len(frontier_fwd) <= len(frontier_bwd):
            new_frontier = set()
            for node in frontier_fwd:
                for code, neighbor in adj.get(node, ()):
                    if neighbor not in parent_fwd:
                        parent_fwd[neighbor] = (node, code)
                        new_frontier.add(neighbor)
                        if neighbor in parent_bwd:
                            meeting_node = neighbor
                            break
                if meeting_node is not None:
                    break
            frontier_fwd = new_frontier
        else:
            new_frontier = set()
            for node in frontier_bwd:
                for code, neighbor in adj.get(node, ()):
                    if neighbor not in parent_bwd:
                        parent_bwd[neighbor] = (node, code)
                        new_frontier.add(neighbor)
                        if neighbor in parent_fwd:
                            meeting_node = neighbor
                            break
                if meeting_node is not None:
                    break
            frontier_bwd = new_frontier

        if meeting_node is not None:
            break

    if meeting_node is None:
        return None

    fwd_codes = []
    node = meeting_node
    while parent_fwd[node] is not None:
        parent, code = parent_fwd[node]
        fwd_codes.append(code)
        node = parent
    fwd_codes.reverse()

    bwd_codes = []
    node = meeting_node
    while parent_bwd[node] is not None:
        parent, code = parent_bwd[node]
        bwd_codes.append(code ^ 1)
        node = parent

    return fwd_codes + bwd_codes


def process_dataset_split(dataset_short, split, qids, entity_to_id, relation_list, adj, max_hops):
    dataset_name = DATASET_NAME_MAP[dataset_short]
    print(f"Carregando {dataset_name}/{split} ...")
    dataset = load_dataset(dataset_name, split=split)

    entries = []
    no_path_qids = []

    for ex in dataset:
        if ex["id"] not in qids:
            continue

        topic_entities = ex["q_entity"]
        answers = ex["a_entity"]
        chains = []

        for q_ent in topic_entities:
            q_id = entity_to_id.get(q_ent)
            if q_id is None:
                continue
            for a_ent in answers:
                a_id = entity_to_id.get(a_ent)
                if a_id is None:
                    continue
                codes = bidirectional_shortest_path(adj, q_id, a_id, max_hops)
                if codes is None:
                    continue
                chains.append([relation_label(c, relation_list) for c in codes])

        if not chains:
            no_path_qids.append(ex["id"])

        entries.append({
            "qid": ex["id"],
            "question": ex["question"],
            "topic_entities": topic_entities,
            "answers": answers,
            "paths": chains,
        })

    return entries, no_path_qids


def write_csv(entries, path):
    """
    Mesmo schema unificado do build_relpaths_dataset.py do MetaQA (qid, question,
    topic_entities, answers, paths). Diferenca: aqui pode haver mais de uma cadeia de
    relacoes por exemplo (respostas diferentes podem exigir caminhos diferentes), entao
    'paths' guarda varias cadeias separadas por ';', cada cadeia com as relacoes
    separadas por '|' (mesmo delimitador interno do MetaQA).
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "question", "topic_entities", "answers", "paths"])
        for e in entries:
            writer.writerow([
                e["qid"], e["question"],
                "|".join(e["topic_entities"]), "|".join(e["answers"]),
                ";".join("|".join(chain) for chain in e["paths"]),
            ])


def parse_dataset_pairs(pairs):
    parsed = []
    for dataset_short, split in pairs:
        if dataset_short not in DATASET_NAME_MAP:
            raise ValueError(
                f"dataset '{dataset_short}' invalido. Opcoes: {list(DATASET_NAME_MAP)}"
            )
        if split not in SPLIT_CHOICES:
            raise ValueError(f"split '{split}' invalido. Opcoes: {SPLIT_CHOICES}")
        parsed.append((dataset_short, split))
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Para cada exemplo qualificado (ver build_merged_subgraph.py), busca o "
                     "caminho mais curto (BFS bidirecional) entre cada topic_entity e cada "
                     "answer no subgrafo global mesclado, e salva o dataset de "
                     "pergunta+relation-paths no formato unificado do projeto (mesmo schema "
                     "CSV do MetaQA)."
    )
    parser.add_argument(
        "--dataset", action="append", nargs=2, metavar=("DATASET", "SPLIT"),
        help="Par dataset/split a processar, ex: --dataset webqsp train. Repetivel. "
             f"Dataset in {list(DATASET_NAME_MAP)}, split in {SPLIT_CHOICES}. "
             "Se omitido, processa todos os 6 combos."
    )
    parser.add_argument(
        "--graph",
        default="src/kgqa/data/kgs/freebase/rog-subgraph/kg/merged_subgraph.tsv",
        help="Caminho do subgrafo global mesclado."
    )
    parser.add_argument(
        "--stats",
        default="src/kgqa/data/kgs/freebase/rog-subgraph/kg/merged_subgraph_stats.json",
        help="Caminho do relatorio com os qids qualificados por dataset/split "
             "(gerado por build_merged_subgraph.py)."
    )
    parser.add_argument(
        "--max_hops", type=int, default=3,
        help="Numero maximo de hops (relacoes) no caminho buscado (default: 3)."
    )
    parser.add_argument(
        "--output_dir_template",
        default="src/kgqa/data/qa/{dataset}/outputs",
        help="Template do diretorio de saida, com {dataset} substituido por webqsp/cwq."
    )
    args = parser.parse_args()

    if args.dataset:
        pairs = parse_dataset_pairs(args.dataset)
    else:
        pairs = [(d, s) for d in DATASET_NAME_MAP for s in SPLIT_CHOICES]

    print(f"Carregando grafo mesclado de {args.graph} ...")
    entity_to_id, relation_list, adj = load_graph(args.graph)
    print(f"  {len(entity_to_id)} entidades, {len(relation_list)} relacoes, "
          f"{sum(len(v) for v in adj.values())} arestas (direta+inversa)")

    with open(args.stats, "r", encoding="utf-8") as f:
        stats = json.load(f)

    for dataset_short, split in pairs:
        key = f"{dataset_short}/{split}"
        qids = set(stats.get(key, {}).get("qids", []))
        if not qids:
            print(f"{dataset_short}/{split}: nenhum qid qualificado, pulando")
            continue

        entries, no_path_qids = process_dataset_split(
            dataset_short, split, qids, entity_to_id, relation_list, adj, args.max_hops
        )

        output_dir = args.output_dir_template.format(dataset=dataset_short)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{dataset_short}_{split}_relpaths.csv")
        write_csv(entries, output_path)

        report_path = os.path.join(output_dir, f"{dataset_short}_{split}_no_path_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": dataset_short,
                "split": split,
                "total_qualifying": len(qids),
                "no_path_count": len(no_path_qids),
                "no_path_qids": no_path_qids,
            }, f, indent=2, ensure_ascii=False)

        print(
            f"{dataset_short}/{split}: {len(entries) - len(no_path_qids)}/{len(entries)} "
            f"exemplos com pelo menos um caminho encontrado | salvos em {output_path}"
        )


if __name__ == "__main__":
    main()
