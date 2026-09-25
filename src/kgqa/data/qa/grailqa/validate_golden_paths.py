import argparse
import json

from SPARQLWrapper import SPARQLWrapper, JSON


FREEBASE_NS = "http://rdf.freebase.com/ns/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate GrailQA gold paths using Virtuoso."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the GrailQA gold paths JSON file."
    )

    parser.add_argument(
        "--endpoint",
        required=True,
        help="Virtuoso SPARQL endpoint."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the validated output JSON file."
    )

    return parser.parse_args()


def create_sparql_client(endpoint):
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)

    return sparql


def query_relation(
    sparql,
    entity,
    relation,
    reverse=False
):
    """
    Queries Virtuoso for all triples matching one
    relation step from the current entity.

    Forward:
        entity --relation--> object

    Reverse:
        subject --relation--> entity
    """

    entity_uri = FREEBASE_NS + entity
    relation_uri = FREEBASE_NS + relation

    if reverse:

        query = f"""
        SELECT ?head ?relation ?tail
        WHERE {{
            ?head <{relation_uri}> <{entity_uri}> .
            BIND(<{relation_uri}> AS ?relation)
            BIND(<{entity_uri}> AS ?tail)
        }}
        """

    else:

        query = f"""
        SELECT ?head ?relation ?tail
        WHERE {{
            <{entity_uri}> <{relation_uri}> ?tail .
            BIND(<{entity_uri}> AS ?head)
            BIND(<{relation_uri}> AS ?relation)
        }}
        """

    sparql.setQuery(query)

    results = sparql.query().convert()

    triples = []

    for result in results["results"]["bindings"]:

        head = result["head"]["value"].replace(
            FREEBASE_NS,
            ""
        )

        relation_name = result["relation"]["value"].replace(
            FREEBASE_NS,
            ""
        )

        tail = result["tail"]["value"].replace(
            FREEBASE_NS,
            ""
        )

        triples.append({
            "head": head,
            "relation": relation_name,
            "tail": tail
        })

    return triples


def expand_path(
    sparql,
    topic_entity,
    relations
):
    """
    Expands a complete relation path from a topic entity.

    Returns all possible complete paths.

    Each result contains the triples used to traverse
    the entire relation path.
    """

    paths = [
        {
            "current_entity": topic_entity,
            "entities": [topic_entity],
            "triples": []
        }
    ]

    for relation in relations:

        reverse = relation.startswith("(R ")

        if reverse:
            relation_name = relation[3:-1]
        else:
            relation_name = relation

        next_paths = []

        for path in paths:

            current_entity = path["current_entity"]

            triples = query_relation(
                sparql=sparql,
                entity=current_entity,
                relation=relation_name,
                reverse=reverse
            )

            for triple in triples:

                next_entity = (
                    triple["head"]
                    if reverse
                    else triple["tail"]
                )

                # Avoid cycles inside this path
                if next_entity in path["entities"]:
                    continue

                next_paths.append({
                    "current_entity": next_entity,
                    "entities": path["entities"] + [
                        next_entity
                    ],
                    "triples": path["triples"] + [
                        triple
                    ]
                })

        paths = next_paths

        if not paths:
            break

    return paths


def validate_dataset(
    dataset,
    sparql
):

    output = []

    empty_results = 0

    for index, example in enumerate(dataset, start=1):

        kg_results = []

        for relations in example["paths"]:

            for topic_entity in example["topic_entities"]:

                paths = expand_path(
                    sparql=sparql,
                    topic_entity=topic_entity,
                    relations=relations
                )

                for path in paths:

                    kg_results.append({
                        "triples": path["triples"]
                    })

        if not kg_results:
            empty_results += 1

        output.append({
            "qid": example["qid"],
            "question": example["question"],
            "topic_entities": example["topic_entities"],
            "paths": example["paths"],
            "answers": example["answers"],
            "kg_results": kg_results
        })

        if index % 100 == 0:
            print(
                f"Processed: {index}/{len(dataset)}"
            )

    print()
    print(
        f"Examples with empty kg_results: "
        f"{empty_results}"
    )

    return output


def save_results(dataset, path):

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )


def main():

    args = parse_args()

    print("Loading gold paths...")

    with open(
        args.input,
        "r",
        encoding="utf-8"
    ) as f:

        dataset = json.load(f)

    print("Connecting to Virtuoso...")

    sparql = create_sparql_client(
        args.endpoint
    )

    print("Validating paths...")

    results = validate_dataset(
        dataset=dataset,
        sparql=sparql
    )

    print("Saving results...")

    save_results(
        dataset=results,
        path=args.output
    )

    print()
    print(f"Examples processed: {len(results)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()