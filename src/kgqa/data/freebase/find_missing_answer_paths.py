import argparse
import json
import os
import time

from datasets import load_dataset
from SPARQLWrapper import SPARQLWrapper, JSON

FREEBASE_NS = "http://rdf.freebase.com/ns/"


def create_sparql_client(endpoint, timeout=60):
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)
    return sparql


def _strip_ns(uri):
    return uri.replace(FREEBASE_NS, "")


def _chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def fetch_neighbors_batch(sparql, entities, batch_size=200, result_limit=50000, max_retries=2):
    """
    Para um lote de entidades, busca todos os vizinhos diretos (1 hop) no Freebase,
    nas duas direções (forward e reverse), em lotes via VALUES.

    Retorna uma lista de tuplas (source_entity, relation, neighbor_entity, direction),
    onde direction é "fwd" (source --relation--> neighbor) ou
    "rev" (neighbor --relation--> source).
    """
    all_results = []

    for chunk in _chunk(entities, batch_size):
        values_clause = " ".join(f"<{FREEBASE_NS}{e}>" for e in chunk)

        query_fwd = f"""
        SELECT ?entity ?relation ?neighbor
        WHERE {{
            VALUES ?entity {{ {values_clause} }}
            ?entity ?relation ?neighbor .
            FILTER(isIRI(?neighbor))
        }}
        LIMIT {result_limit}
        """

        query_rev = f"""
        SELECT ?entity ?relation ?neighbor
        WHERE {{
            VALUES ?entity {{ {values_clause} }}
            ?neighbor ?relation ?entity .
            FILTER(isIRI(?neighbor))
        }}
        LIMIT {result_limit}
        """

        for query, direction in [(query_fwd, "fwd"), (query_rev, "rev")]:
            for attempt in range(max_retries + 1):
                try:
                    sparql.setQuery(query)
                    results = sparql.query().convert()
                    break
                except Exception as e:
                    if attempt < max_retries:
                        print(f"    [aviso] erro na consulta ({direction}), tentativa {attempt + 1}: {e}")
                        time.sleep(2)
                    else:
                        print(f"    [erro] falhou apos {max_retries + 1} tentativas ({direction}): {e}")
                        results = {"results": {"bindings": []}}

            for r in results["results"]["bindings"]:
                entity = _strip_ns(r["entity"]["value"])
                relation = _strip_ns(r["relation"]["value"])
                neighbor = _strip_ns(r["neighbor"]["value"])
                all_results.append((entity, relation, neighbor, direction))

    return all_results


def find_paths_for_example(sparql, topic_entities, missing_answers, max_hops=3,
                            max_frontier=20000, batch_size=200):
    """
    Busca multi-fonte (BFS) a partir das topic_entities, procurando caminhos
    ate cada entidade em missing_answers, ate max_hops.

    Retorna um dict {answer_entity: {"hops": n, "concrete_path": [triplas]}}
    apenas para as respostas encontradas.
    """
    topic_set = set(topic_entities)
    targets = set(missing_answers) - topic_set  # ja e a topic entity, nada a fazer
    found = {}

    # parent[entity] = (parent_entity, relation, direction)
    # direction indica a direcao da aresta entre parent_entity e entity
    parent = {}
    visited = set(topic_set)
    frontier = list(topic_set)

    for hop in range(1, max_hops + 1):
        if not (targets - found.keys()):
            break
        if not frontier:
            break

        if len(frontier) > max_frontier:
            print(f"    [aviso] frontier truncado de {len(frontier)} para {max_frontier} no hop {hop}")
            frontier = frontier[:max_frontier]

        neighbors = fetch_neighbors_batch(sparql, frontier, batch_size=batch_size)

        next_frontier = []
        for entity, relation, neighbor, direction in neighbors:
            if neighbor in visited:
                continue
            if neighbor not in parent:
                parent[neighbor] = (entity, relation, direction)
                visited.add(neighbor)
                next_frontier.append(neighbor)

            if neighbor in targets and neighbor not in found:
                found[neighbor] = hop

        frontier = next_frontier

    # Reconstroi os caminhos concretos (lista de triplas) para cada resposta encontrada
    results = {}
    for answer, hop in found.items():
        path = []
        current = answer
        while current not in topic_set:
            p_entity, relation, direction = parent[current]
            if direction == "fwd":
                # p_entity --relation--> current
                path.append({"head": p_entity, "relation": relation, "tail": current})
            else:
                # current --relation--> p_entity
                path.append({"head": current, "relation": relation, "tail": p_entity})
            current = p_entity
        path.reverse()
        results[answer] = {"hops": hop, "concrete_path": path}

    return results


def load_examples_by_qid(dataset_name, split):
    """Carrega o dataset do HuggingFace e indexa por qid, para recuperar topic_entities."""
    dataset = load_dataset(dataset_name, split=split)
    return {ex["id"]: ex for ex in dataset}


def load_existing_output(output_path):
    """Carrega resultados ja processados (para retomar execucoes interrompidas)."""
    if not os.path.exists(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_report(report_path, sparql, output_path, max_hops, max_frontier,
                    batch_size, limit=None):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    output = load_existing_output(output_path)

    for key, problems in report["problem_examples"].items():
        if not problems:
            continue

        dataset_name, split = key.rsplit("/", 1)

        already_done = {r["qid"] for r in output.get(key, [])}

        pending = [p for p in problems if p["qid"] not in already_done]
        if limit is not None:
            pending = pending[:limit]

        if not pending:
            print(f"{key}: nada a processar (ja concluido ou fora do --limit)")
            continue

        print(f"Carregando {dataset_name}/{split} para localizar topic_entities...")
        examples_by_qid = load_examples_by_qid(dataset_name, split)

        split_results = output.get(key, [])

        for i, problem in enumerate(pending, start=1):
            qid = problem["qid"]
            missing = problem["missing_answers"]
            example = examples_by_qid.get(qid)

            if example is None:
                print(f"  [aviso] qid {qid} nao encontrado no dataset, pulando")
                continue

            topic_entities = example["q_entity"]

            found_paths = find_paths_for_example(
                sparql, topic_entities, missing,
                max_hops=max_hops, max_frontier=max_frontier, batch_size=batch_size,
            )

            split_results.append({
                "qid": qid,
                "topic_entities": topic_entities,
                "missing_answers": missing,
                "found": found_paths,
                "still_missing": [a for a in missing if a not in found_paths],
            })

            if i % 5 == 0 or i == len(pending):
                print(f"  {key}: {i}/{len(pending)} processados neste lote "
                      f"({len(split_results)}/{len(problems)} no total)")

                output[key] = split_results
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)

        output[key] = split_results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    return output


def print_summary(output):
    print("\n" + "=" * 70)
    print("Resumo")
    print("=" * 70)
    for key, results in output.items():
        total_missing = sum(len(r["missing_answers"]) for r in results)
        total_found = sum(len(r["found"]) for r in results)
        exemplos_totalmente_recuperados = sum(
            1 for r in results if len(r["still_missing"]) == 0
        )
        print(
            f"{key:<35} exemplos={len(results):<6} "
            f"respostas_encontradas={total_found}/{total_missing} "
            f"exemplos_100%_recuperados={exemplos_totalmente_recuperados}/{len(results)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Busca caminhos no Freebase completo (via Virtuoso) para as respostas "
                     "que nao apareceram nos subgrafos pre-extraidos do RoG."
    )
    parser.add_argument(
        "--report", default="src/kgqa/data/freebase/outputs/answer_coverage_report.json",
        help="Caminho do relatorio gerado por check_answer_coverage.py."
    )
    parser.add_argument(
        "--endpoint", required=True,
        help="Endpoint SPARQL do Virtuoso (ex: http://localhost:8891/sparql)."
    )
    parser.add_argument(
        "--output", default="src/kgqa/data/freebase/outputs/recovered_answer_paths.json",
        help="Caminho do arquivo de saida."
    )
    parser.add_argument(
        "--max_hops", type=int, default=3,
        help="Numero maximo de hops a explorar a partir das topic entities (default: 3)."
    )
    parser.add_argument(
        "--max_frontier", type=int, default=20000,
        help="Numero maximo de entidades exploradas por hop, para evitar explosao "
             "em entidades de grau muito alto (default: 20000)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=200,
        help="Numero de entidades agrupadas por consulta SPARQL via VALUES (default: 200)."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Processa no maximo N exemplos pendentes por dataset/split (para testes rapidos)."
    )
    args = parser.parse_args()

    print("Conectando ao Virtuoso...")
    sparql = create_sparql_client(args.endpoint)

    output = process_report(
        report_path=args.report,
        sparql=sparql,
        output_path=args.output,
        max_hops=args.max_hops,
        max_frontier=args.max_frontier,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    print_summary(output)
    print(f"\nResultado salvo em: {args.output}")


if __name__ == "__main__":
    main()
