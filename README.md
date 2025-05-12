# AlignRuScore: Adapting AlignScore to Russian Language

**AlignRuScore** is a project focused on adapting the AlignScore metric for evaluating factual consistency to the Russian language. This work provides a unified evaluation framework covering natural language inference, fact verification, paraphrase detection, semantic textual similarity, question answering, and information retrieval for Russian.

<!-- This repository contains the code, links to translated datasets, and model checkpoints for AlignRuScore. -->

## Abstract

Ensuring factual consistency in generated text is crucial for reliable natural language processing applications. We introduce AlignRuScore, a comprehensive adaptation of the AlignScore metric [1] for Russian. This unified evaluation covers a wide array of NLP tasks. We compiled and translated over 118,000 examples — combining major English benchmarks with Russian-native datasets (RuFacts [2], RuSTS Benchmark) — and fine-tuned a RuBERT-based [3] alignment model with task-specific classification and regression heads. AlignRuScore demonstrates strong performance on various tasks, laying the groundwork for robust multilingual factual consistency evaluation. We release our translated corpora, model checkpoints, and code to support further research.

## Background: AlignScore Main Idea

AlignScore is a metric for evaluating the factual consistency of generated text by assessing the alignment of information between a claim and its context. It uses a unified text-to-text information alignment function, trained on a diverse set of data sources from various NLP tasks, to estimate an alignment score.

*(Consider adding the diagram from slide 3 of your presentation here, illustrating the unified alignment function concept)*
`![AlignScore Main Idea Diagram](path_to_your_image_slide_3.png)`
![AlignScore Main Idea Diagram](path_to_your_image_slide_3.png)

AlignScore handles long texts by splitting the context into coarse-grained chunks (approx. 350 tokens) and the claim into fine-grained sentences. It then aggregates the alignment scores between context-chunks and claim-sentences to produce a final factual consistency score.

*(Consider adding the diagram from slide 4 of your presentation here, illustrating context/claim splitting)*
`![AlignScore Context/Claim Splitting Diagram](path_to_your_image_slide_4.png)`

## AlignRuScore: Methodology

Our methodology adapts the original AlignScore framework to the Russian language.

### 1. Data Collection and Translation
We constructed a diverse Russian training corpus by:
*   Translating subsets (up to 10,000 examples each where applicable) of the English datasets used in the original AlignScore paper (covering NLI, Fact Verification, Paraphrase, QA, STS). Machine translation was primarily performed using Yandex Translate.
*   Supplementing with Russian-native datasets:
    *   **RuFacts**: For paraphrase/fact verification.
    *   **RuSTS Benchmark**: For semantic textual similarity.
The final unified corpus comprises over 118,900 training examples.

### 2. Model Training
*   **Base Model**: RuBERT-base (180M parameters).
*   **Training Approach**: Unified multi-task learning with task-specific heads for:
    *   3-way classification (ALIGNED, CONTRADICT, NEUTRAL)
    *   Binary classification (ALIGNED or NOT-ALIGNED)
    *   Regression (similarity score between 0 and 1)
*   **Key Hyperparameters**:
    *   Optimizer: AdamW
    *   Learning Rate: 1e-5
    *   Batch Size: 12
    *   Epochs: 3
    *   GPU: 1/4 A100 (Training time ~3.5 hours)

A small feed-forward network is trained for each task type simultaneously, allowing the embedding space and heads to handle alignment for all NLP alignment tasks.

## Key Results

AlignRuScore was evaluated on held-out test portions of Russian datasets.

### Classification Tasks:
*   **3-Way Classification (Entailment, Fact Verification):**
    *   SNLI: F1 = 0.743, Accuracy = 0.743
    *   FEVER: F1 = 0.814, Accuracy = 0.814
    *   MultiNLI: F1 = 0.674
    *   ANLI: F1 = 0.692
    *   VitaminC: F1 = 0.596
*   **Binary Classification (Paraphrase, QA, IR, Document NLI):**
    *   QQP: F1 = 0.768, ROC AUC = 0.879
    *   DocNLI: F1 = 0.847, ROC AUC = 0.897
    *   RuFact: F1 = 0.716, ROC AUC = 0.732
    *   PAWS: F1 = 0.728
    *   MS MARCO: F1 = 0.289 (Precision = 0.201, Recall = 0.520)
    *   RACE: F1 = 0.400 (Precision = 0.250, Recall = 0.998)

### Regression Tasks (Semantic Textual Similarity):
*   SICK (translated): R² = 0.857, MSE = 0.228
*   RuSTS Benchmark: R² = 0.557, MSE = 0.954

### LLM Evaluation:
AlignRuScore was used to evaluate outputs from Gemini 1.5 Flash on a subset of 200 samples from the IlyaGusev/gazeta summarization dataset, yielding an average factual-consistency score of **0.7285 ± 0.0639**.

These results demonstrate that a unified alignment metric can be successfully ported to Russian. Performance is strong in controlled entailment and paraphrase detection, with areas for improvement in open-domain QA and Russian STS.

## Released Assets

*   **Code**: This GitHub repository.
*   **Translated Datasets**: [MilyaShams/AlignScore_russian_datasets](https://huggingface.co/datasets/MilyaShams/AlignScore_russian_datasets)
*   **Model Checkpoint (RuBERT-base fine-tuned for AlignRuScore)**: [CatFr0g/ruAlignScore](https://huggingface.co/CatFr0g/ruAlignScore)

## Future Plans

1.  Incorporate additional Russian-native datasets, particularly for summarization and dialogue consistency.
2.  Explore architecture variants, such as multilingual transformer backbones and task-adaptive adapters.
3.  Evaluate AlignRuScore in downstream applications, including automated fact-checking and evaluation of Russian-language generative models.

## Links and References

*   **[1] AlignScore Paper**: Zha, Y., Yang, Y., Li, R., & Hu, Z. (2023). AlignScore: Evaluating Factual Consistency with A Unified Alignment Function. arXiv preprint arXiv:2305.16739. ([https://arxiv.org/abs/2305.16739](https://arxiv.org/abs/2305.16739))
*   **[2] RuFacts Dataset**: akozlova/RuFacts ([https://huggingface.co/datasets/akozlova/RuFacts](https://huggingface.co/datasets/akozlova/RuFacts)) / SberDevices. (2023). Fact-checking benchmark for the Russian Large Language Models. ([Paper Link from Presentation](https://www.example.com/sberdevices_fact_checking_paper)) *(Please replace with actual link if available)*
*   **[3] RuBERT Model**: DeepPavlov/rubert-base-cased ([https://huggingface.co/DeepPavlov/rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased)) / Kuratov, Y., & Arkhipov, M. (2019). Adaptation of deep bidirectional multilingual transformers for russian language. arXiv preprint arXiv:1905.07213.
*   **RuSTS Benchmark STS Dataset**: ai-forever/ru-stsbenchmark-sts ([https://huggingface.co/datasets/ai-forever/ru-stsbenchmark-sts](https://huggingface.co/datasets/ai-forever/ru-stsbenchmark-sts))
*   **This Project's Paper (Preprint/Draft)**: Zimin, M., & Shamsutdinova, M. (2025). AlignRuScore: Adapting AlignScore to Russian Language. *(Link to your ArXiv preprint or paper once available)*

## Citation

If you use AlignRuScore or the translated datasets in your research, please consider citing our work:

```bibtex
@misc{zimin_shamsutdinova_alignruscore_2025,
  title={AlignRuScore: Adapting AlignScore to Russian Language},
  author={Mikhail Zimin and Milyausha Shamsutdinova},
  year={2025},
  howpublished={GitHub repository and NLP project paper},
  note={URL to your paper/repo}
}