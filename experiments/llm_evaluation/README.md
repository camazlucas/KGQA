# LLM Evaluation for KGQA

## 1. Overview

This experiment evaluates Large Language Models (LLMs) in a Knowledge Graph Question Answering (KGQA) task under a zero-shot setting.

The experiment investigates whether providing structured information from a knowledge graph as context can improve the ability of causal language models to answer questions. Two forms of knowledge graph context are evaluated:

- **Triple-based context:** the model receives a set of knowledge graph triples related to the question.
- **Path-based context:** the model receives concrete paths connecting entities through relations in the knowledge graph.

Both approaches use the same questions from the GrailQA dataset, allowing a direct comparison between providing isolated triples and providing structured paths representing multi-hop connections in the knowledge graph.

The experiment also measures answer quality, inference time, and GPU memory consumption for each evaluated model.

## 2. Experimental Objective

The main objective of this experiment is to evaluate the ability of causal LLMs to answer KGQA questions when provided with information retrieved from a knowledge graph as contextual evidence.

The experiment investigates two main research questions:

1. **Triple-based vs. path-based context:** whether representing the retrieved knowledge as connected paths provides an advantage over presenting the same information as independent knowledge graph triples.

2. **LLM performance and computational cost:** how different causal LLMs perform in terms of answer accuracy, inference time, and GPU memory consumption when applied to KGQA.

The evaluation is conducted under a **zero-shot setting**, without task-specific fine-tuning. This allows the experiment to isolate the effect of the knowledge graph representation provided as context and the inherent reasoning capabilities of the evaluated language models.

## 3. Experimental Setup

The experiment evaluates causal LLMs using the GrailQA dataset and contextual information retrieved from the corresponding knowledge graph.

For each question, the model receives a prompt containing:

- the natural language question;
- knowledge graph information associated with the question;
- an instruction to identify the entity that answers the question.

The models are evaluated without task-specific fine-tuning. Each model processes the questions independently, and its generated response is compared against the gold answer provided by GrailQA.

Two experimental conditions are considered:

1. **Triple-based evaluation**, in which the context consists of knowledge graph triples.
2. **Path-based evaluation**, in which the context consists of concrete knowledge graph paths.

The experiments are executed independently for each model and each context representation. Inference time and GPU memory consumption are recorded alongside the predicted answers.

### 3.1 Dataset

The experiments use the GrailQA dataset, a benchmark for Knowledge Graph Question Answering over Freebase.

The evaluation uses a filtered subset of GrailQA containing **962 questions**. For each question, the dataset contains the corresponding topic entity, gold answer, and knowledge graph information required to construct the experimental context.

Two derived datasets are used:

- `grailqa_triples_text.json`: contains the knowledge graph triples converted to textual representations for the triple-based experiment.
- `grailqa_paths_text.json`: contains concrete knowledge graph paths converted to textual representations for the path-based experiment.

The path-based dataset applies a maximum limit of **150 concrete paths per question**. This restriction was introduced to prevent excessively large contexts that could exceed the context window or GPU memory capacity of the evaluated LLMs.

### 3.2 Triple-based Evaluation

In the triple-based experiment, the knowledge graph context is provided as a collection of textualized triples associated with each question.

Each triple follows the structure:

Head | Relation | Tail

The model is instructed to use the triples as contextual evidence and determine which entity answers the question. The prompt explicitly identifies the semantic role of each component and asks the model to return only the answer, without additional explanations or reasoning.

The triple-based evaluation is implemented by `run_experiment_triples.py`.

The input dataset is:

`src/kgqa/data/grailqa/grailqa_triples_text.json`

Results are stored in:

`results/llm_evaluation/triples/`

This configuration establishes the baseline for evaluating whether a causal LLM can identify the correct answer from a set of retrieved knowledge graph facts without explicitly representing the connections between multiple facts as paths.### 3.3 Path-based Evaluation

### 3.3 Path-based Evaluation

In the path-based experiment, the knowledge graph context is represented as concrete paths connecting entities through one or more relations.

Each concrete path is represented as an ordered sequence of entities and relations. For example, a one-hop path is represented as:

Entity A → Relation → Entity B

A multi-hop path extends this structure by including intermediate entities and relations. For example, a three-hop path can be represented as:

Entity A → Relation 1 → Entity B → Relation 2 → Entity C → Relation 3 → Entity D

The paths are constructed from the gold/retrieved relation paths and instantiated with concrete entities from the knowledge graph. This representation explicitly preserves the connectivity between the entities involved in the reasoning chain.

The model is instructed to use the provided paths as contextual evidence and identify the entity that answers the question. As in the triple-based experiment, the model is instructed to return only the answer without explaining its reasoning.

The path-based evaluation is implemented by `run_experiment_paths.py`.

The input dataset is:

`src/kgqa/data/grailqa/grailqa_paths_text.json`

Results are stored in:

`results/llm_evaluation/paths/`

To control the size of the input context, the path-based dataset is limited to a maximum of **150 concrete paths per question**. This limit prevents questions with thousands of candidate paths from producing excessively large prompts and causing context-length or GPU memory issues.

### 3.4 Zero-Shot Setting

All LLM evaluations are performed under a **zero-shot setting**. The models are used without task-specific fine-tuning on the GrailQA dataset or on the KGQA task.

For each question, the model receives a prompt containing the question and the corresponding knowledge graph context. The model then generates an answer based solely on the information provided in the prompt and its pretrained knowledge.

No demonstrations, labeled examples, or task-specific training instances are included in the prompts.

The same evaluation procedure is applied across the different models and both context representations, ensuring that the comparison focuses on the effect of the knowledge graph representation and the capabilities of the evaluated LLMs.

The generated responses are stored together with the question identifier, gold answer, inference time, and other experimental metadata for subsequent evaluation.

## 4. Models

The experiment evaluates a set of causal language models with different architectures, parameter scales, and instruction-tuning characteristics.

The models are evaluated independently under both the triple-based and path-based conditions. Each model receives the same questions and corresponding knowledge graph context for a given experimental condition.

The models are grouped into two categories:

- **Evaluated models:** models included in the final experimental runs.
- **Discarded models:** models initially considered but excluded from the experiments due to unsuitable answer quality, computational limitations, or compatibility issues.

### 4.1 Evaluated Models

The following models are included in the zero-shot evaluation:

| Model | Parameters |
|---|---:|
| Qwen 0.5B | 0.5B |
| Llama 3.2 3B | 3B |
| Llama 2 Chat 7B | 7B |
| Qwen 2.5 7B | 7B |
| Llama 8B | 8B |
| Llama 3 8B | 8B |

The models are executed using the same experimental pipeline, with the model-specific loading configuration handled by `causal/loader.py`.

### 4.2 Discarded Models

Several models were initially considered for the zero-shot evaluation but were excluded from the final set due to inadequate answer quality, excessive computational requirements, or compatibility problems with the experimental environment.

| Model | Reason for exclusion |
|---|---|
| DeepSeek Coder 1.3B | Generated answers were frequently verbose, refused to answer some KGQA questions, or produced programming-oriented responses instead of directly answering the question. |
| DeepSeek Coder 6.7B | Generated excessively verbose responses and sometimes interpreted the task as a programming problem rather than KGQA. Inference was also considerably slower. |
| DeepSeek-R1-Distill-Qwen-1.5B | Generated long reasoning-oriented responses instead of returning only the requested entity, making it unsuitable for the intended answer format in the zero-shot setting. |
| DeepSeek-LLM-7B-Chat | Could not be evaluated due to GPU out-of-memory (OOM) errors during model execution. |
| Ministral 3 3B | Could not be evaluated due to an incompatibility between the model's fine-grained FP8 kernel and the installed PyTorch version. |
| Ministral 3 8B Reasoning | Could not be loaded on the available RTX 4090 due to GPU out-of-memory (OOM) during model loading. |
| Qwen-14B | Excluded due to computational/resource constraints for the available GPU. |
| Alpaca-7B | Excluded from the final evaluation set due to unsuitable performance in preliminary testing. |

## 5. Dataset Preparation

The GrailQA data used in the experiment is preprocessed to generate the two knowledge graph context representations required by the evaluation: triples and concrete paths.

The preprocessing preserves the original question identifiers, questions, topic entities, and gold answers, while adding the knowledge graph information required by each experimental condition.

Two textual datasets are generated:

- `grailqa_triples_text.json`: textual representation of the knowledge graph triples associated with each question.
- `grailqa_paths_text.json`: textual representation of the concrete knowledge graph paths associated with each question.

The resulting datasets are used directly by the corresponding experiment scripts, ensuring that the triple-based and path-based evaluations use the same underlying questions and gold answers.

The path-based dataset additionally applies a maximum of **150 concrete paths per question** to control the size of the context provided to the language models.

### 5.1 Triple Representation

For the triple-based evaluation, the knowledge graph information is represented as individual triples in the following form:

Head | Relation | Tail

For example:

```text
The Illusion | theater.play.productions | The Illusion
```
Each triple represents a direct relationship between two entities. The textual representation is constructed from the entities and relations retrieved from the knowledge graph.

The resulting triples are provided to the LLM as a collection of contextual facts. The model must determine which entity in the provided triples corresponds to the answer to the question.

The textualized triples are stored in:

src/kgqa/data/grailqa/grailqa_triples_text.json

### 5.2 Path Representation

For the path-based evaluation, the knowledge graph information is represented as **concrete paths** connecting the topic entity to candidate answer entities.

Each path is represented as an ordered sequence alternating between entities and relations. Reverse relations are explicitly marked with the `(R ...)` notation.

For example, a one-hop path is represented as:

```text
Avanti Monza
(R bicycles.bicycle_type.bicycle_models_of_this_type)
Road bicycle
```
For multi-hop questions, intermediate entities and relations are included in the same sequence. For example:
```
flof.com.ar
(R internet.api.site)
XML
internet.api.site
StopFinder
(R internet.website_owner.websites_owned)
StopFinder Ltd.
```
This representation preserves the connectivity between successive entities and explicitly exposes the reasoning chain required to reach the candidate answer.

Unlike the triple-based representation, where each fact is presented independently, the path-based representation provides the LLM with the complete sequence of relations connecting the entities involved in the answer.

The textualized paths are stored in:

src/kgqa/data/grailqa/grailqa_paths_text.json

### 5.3 Path Limit

The number of concrete paths associated with each question can vary substantially in GrailQA. While most questions contain a small number of paths, a small number of questions contain hundreds or thousands of candidate paths.

An analysis of the 962 questions used in the experiment showed the following distribution:

| Number of paths | Questions |
|---|---:|
| 1 | 578 |
| 2–5 | 208 |
| 6–10 | 44 |
| 11–50 | 75 |
| 51–100 | 20 |
| 101–500 | 22 |
| 501–1000 | 5 |
| 1001–5000 | 4 |
| >5000 | 6 |

The dataset has an average of **61.67 paths per question**, with a maximum of **9,992 paths**.

These extreme cases can result in excessively large prompts. During preliminary experiments, contexts containing a very large number of paths exceeded the available context length and GPU memory capacity, resulting in out-of-memory errors.

To control the context size while retaining the majority of the available path information, a maximum of **150 concrete paths per question** was adopted for the path-based experiment.

Questions containing more than 150 paths are therefore restricted to the first 150 concrete paths in the prepared dataset.

## 6. Prompts

The two experimental conditions use separate prompts designed to provide the corresponding knowledge graph representation as contextual evidence.

The prompts follow the same general structure and differ only in the type of knowledge graph information provided to the model. This ensures that the comparison between triples and paths is focused on the representation of the contextual information rather than on substantially different instructions.

In both cases, the model receives the knowledge graph context followed by the natural language question and is instructed to return only the answer without explaining its reasoning.

### 6.1 Triple Prompt

The triple-based prompt provides the textualized knowledge graph triples and explicitly describes their structure as:

Head | Relation | Tail

The model is instructed to use the triples to determine which entity answers the question and to return only the answer.

The prompt follows the structure:

```
text
Use the following structured knowledge graph triples as context to answer the question.

Knowledge Graph triples:

{kg_triples}

Each triple is represented as:

Head | Relation | Tail

The relation describes the relationship from Head to Tail.

Use the triples to determine which entity, Head or Tail, answers the question.

Question:

{question}

Return only the answer.

Do not explain your reasoning.
```

The prompt is constructed by `build_triples_prompt()` in the triple-based experiment.

### 6.2 Path Prompt

The path-based prompt provides the textualized concrete knowledge graph paths associated with the question.

The model is instructed to interpret each path as a sequence of connected entities and relations and use these paths to determine which entity answers the question.

The prompt follows the structure:

```text
Use the following knowledge graph paths as context to answer the question.

Knowledge Graph paths:

{kg_paths}

Each path represents a sequence of entities and relations connected in the knowledge graph.

Use the paths to determine which entity answers the question.

Question:

{question}

Return only the answer.

Do not explain your reasoning.
```
The prompt is constructed by `build_paths_prompt()` in the path-based experiment.

## 7. Execution

The experiments can be executed either individually for a specific model or sequentially for all models included in the evaluation.

Two independent experiment scripts are provided:

- `run_experiment_triples.py`: executes the triple-based evaluation.
- `run_experiment_paths.py`: executes the path-based evaluation.

Separate shell scripts are provided to execute all models for each experimental condition:

- `run_all_causal_triples.sh`
- `run_all_causal_paths.sh`

All experiments use the same model loading and inference infrastructure, while the experiment-specific script determines which knowledge graph representation is provided to the model.

### 7.1 Single Model

A single model can be evaluated directly using the corresponding experiment script.

For the triple-based condition:

```bash
python -m experiments.llm_evaluation.run_experiment_triples \
    --model <model> \
    --dataset src/kgqa/data/grailqa/grailqa_triples_text.json \
    --results-dir results/llm_evaluation/triples
```
For the path-based condition:

```bash
python -m experiments.llm_evaluation.run_experiment_paths \
    --model <model> \
    --dataset src/kgqa/data/grailqa/grailqa_paths_text.json \
    --results-dir results/llm_evaluation/paths
```

The `--model` argument specifies the model configuration used by the experiment. The `--dataset` argument specifies the corresponding textualized knowledge graph dataset, while `--results-dir` defines the directory in which the generated results are stored.

### 7.2 All Models

All models for a given experimental condition can be executed sequentially using the corresponding shell script.

For the triple-based evaluation:

```bash
bash experiments/llm_evaluation/run_all_causal_triples.sh
```

For the path-based evaluation:
```
bash experiments/llm_evaluation/run_all_causal_paths.sh
```

Each shell script iterates over the list of configured models and executes the corresponding Python experiment script for each model.

The output of each model is redirected to an individual log file under:

`results/llm_evaluation/logs/`

The main execution status is also recorded in:

`run_all_causal_triples.log`
`run_all_causal_paths.log`

If a model terminates with an error, the shell script records the corresponding exit status and continues with the next model.

### 7.3 Background Execution with `nohup`

For long-running experiments on the laboratory server, the shell scripts can be executed in the background using `nohup`. This allows the experiment to continue running after the SSH session is closed.

For the triple-based evaluation:

```
bash
nohup bash experiments/llm_evaluation/run_all_causal_triples.sh \
    > results/llm_evaluation/logs/run_all_causal_triples.log 2>&1 &
```

For the path-based evaluation:

```
bash
nohup bash experiments/llm_evaluation/run_all_causal_paths.sh \
    > results/llm_evaluation/logs/run_all_causal_paths.log 2>&1 &
```

The execution can be monitored using:

```
bash
ps aux | grep run_all_causal
```

The main log can be followed in real time with:

```
bash
tail -f results/llm_evaluation/logs/run_all_causal_paths.log
```

Individual model logs can also be inspected separately under:

`results/llm_evaluation/logs/`

This approach is used for experiments that require extended execution time and should not depend on an active SSH connection.

## 8. Evaluation Metrics

The generated answers are evaluated against the gold answers provided by GrailQA.

The experiment records both **answer quality** and **computational cost**.

### Answer Quality

The following metrics are used:

- **Exact Match (EM):** measures whether the predicted answer exactly matches a gold answer.
- **F1-score:** measures the token-level overlap between the predicted answer and the gold answer, combining precision and recall.
- **Hits@1:** measures whether the correct answer is included as the top-ranked prediction.

### Computational Cost

The following measurements are recorded during inference:

- **Total inference time:** total time required to process all questions.
- **Average time per question:** total inference time divided by the number of evaluated questions.
- **GPU peak memory allocated:** maximum GPU memory allocated by PyTorch during execution.
- **GPU peak memory reserved:** maximum GPU memory reserved by PyTorch during execution.
- **GPU information:** GPU model and total available GPU memory.

These measurements allow the experiment to compare not only the effectiveness of the different LLMs and knowledge graph representations, but also their computational requirements.

## 9. Resource Monitoring

The computational resources used during inference are monitored to characterize the cost of each model.

GPU memory consumption is measured using PyTorch during the execution of each experiment. The following information is recorded in the results metadata:

- GPU model.
- Total GPU memory.
- GPU memory allocated after inference.
- GPU memory reserved by PyTorch after inference.
- Peak GPU memory allocated during execution.
- Peak GPU memory reserved during execution.

The laboratory experiments are performed on an NVIDIA GeForce RTX 4090 with approximately 24 GB of GPU memory.

GPU utilization can also be monitored externally using `nvidia-smi`:

```
bash
nvidia-smi
```

Running processes can be inspected with:

```
bash
nvidia-smi
```

For long-running experiments, the process status can be checked using:

```
bash
ps aux | grep run_experiment
```

These measurements are used to identify computational limitations and compare the resource requirements of the evaluated models.

## 10. Results

The results of the experiments are stored separately according to the knowledge graph context representation:

- `results/llm_evaluation/triples/`: results from the triple-based evaluation.
- `results/llm_evaluation/paths/`: results from the path-based evaluation.
- `results/llm_evaluation/logs/`: execution logs for the evaluated models.

Each experiment generates a JSON file containing the experimental metadata and the predictions produced by the model.

The metadata includes information such as:

- Model name.
- Dataset.
- Number of evaluated questions.
- Total inference time.
- Average inference time per question.
- GPU information.
- GPU memory consumption.

Each prediction contains the question identifier, question, knowledge graph context, generated response, gold answer, and inference time for the individual question.

The final results are evaluated using the metrics described in Section 8. Results from the triple-based and path-based conditions are compared to determine whether representing the knowledge graph as connected paths improves answer quality relative to independent triples.

The numerical results and comparative analysis will be added after the complete execution of the experimental runs.

## 11. Failed and Discarded Models

Several models were evaluated during the preliminary stages of the experiment but were not included in the final evaluation set.

The exclusion criteria were:

- inadequate answer quality in the zero-shot KGQA task;
- excessive GPU memory requirements;
- incompatibility with the available software environment;
- or behavior inconsistent with the required answer format.

The models and the corresponding reasons for exclusion are documented below.

| Model | Reason for exclusion |
|---|---|
| DeepSeek Coder 1.3B | Frequently generated programming-oriented responses or failed to answer the KGQA question directly. |
| DeepSeek Coder 6.7B | Frequently interpreted the task as a programming problem and produced excessively verbose responses. |
| DeepSeek-R1-Distill-Qwen-1.5B | Generated long reasoning-oriented responses instead of returning only the requested answer. |
| DeepSeek-LLM-7B-Chat | Could not be evaluated due to GPU out-of-memory (OOM) errors. |
| Ministral 3 3B | Could not be evaluated due to incompatibility between the model's fine-grained FP8 kernel and the installed PyTorch version. |
| Ministral 3 8B Reasoning | Could not be loaded due to GPU out-of-memory (OOM) during model loading. |
| Qwen-14B | Excluded due to computational/resource constraints on the available GPU. |
| Alpaca-7B | Excluded based on preliminary evaluation results. |

These preliminary tests were used to determine which models were technically feasible and sufficiently suitable for the final zero-shot comparison.

## 12. Experimental Notes

The experiment was developed iteratively, with preliminary tests used to identify suitable models, verify the inference pipeline, and determine appropriate context-size constraints.

The main experimental decisions and observations are summarized below.

### Context Representation

Two independent evaluation pipelines were implemented:

- Triple-based evaluation, using `run_experiment_triples.py`.
- Path-based evaluation, using `run_experiment_paths.py`.

The two approaches use the same questions and gold answers, while differing in how the knowledge graph evidence is presented to the LLM.

### Context Size

The analysis of the path-based dataset showed a highly uneven distribution of concrete paths. Most questions contain only a small number of paths, while a small number contain hundreds or thousands of paths.

The maximum observed number of concrete paths was **9,992**, resulting in contexts large enough to exceed the available model context window and GPU memory.

A maximum of **150 concrete paths per question** was therefore adopted for the path-based dataset.

### Preliminary Model Tests

Several models were tested before defining the final model set. These tests revealed that model size alone does not guarantee suitable performance for the KGQA task. Some smaller models produced answers in an inappropriate format, while some larger models could not be executed within the available GPU memory.

The preliminary tests also showed that models trained primarily for code generation may interpret the KGQA prompt as a programming task rather than directly answering the question.

### Execution Environment

The experiments are executed on an NVIDIA GeForce RTX 4090 with approximately 24 GB of GPU memory.

Long-running experiments on the laboratory server are executed using `nohup`, allowing the processes to continue after the SSH session is disconnected.

Individual model logs are maintained to facilitate the identification of inference errors, resource limitations, and unexpected model behavior.

### Reproducibility

The experiment configuration, preprocessing scripts, model execution scripts, and shell scripts are maintained in the repository.

The experimental results are stored separately for the triple-based and path-based conditions, allowing the two approaches to be evaluated and compared independently.