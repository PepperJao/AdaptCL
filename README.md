# AdaptCL: Adaptive Continual Learning for Tackling Heterogeneity in Sequential Datasets

This repository contains the code and resources related to the research paper titled "AdaptCL: Adaptive Continual Learning for Tackling Heterogeneity in Sequential Datasets."

## Abstract

Managing heterogeneous datasets that vary in complexity, size, and similarity in continual learning presents a significant challenge. Task-agnostic continual learning is necessary to address this challenge, as datasets with varying similarity pose difficulties in distinguishing task boundaries. Conventional task-agnostic continual learning practices typically rely on rehearsal or regularization techniques. However, rehearsal methods may struggle with varying dataset sizes and regulating the importance of old and new data due to rigid buffer sizes. Meanwhile, regularization methods apply generic constraints to promote generalization but can hinder performance when dealing with dissimilar datasets lacking shared features, necessitating a more adaptive approach.

In this paper, we propose AdaptCL, a novel adaptive continual learning method to tackle heterogeneity in sequential datasets. AdaptCL employs fine-grained data-driven pruning to adapt to variations in data complexity and dataset size. It also utilizes task-agnostic parameter isolation to mitigate the impact of varying degrees of catastrophic forgetting caused by differences in data similarity. Through a two-pronged case study approach, we evaluate AdaptCL on both datasets of MNIST Variants and DomainNet, as well as datasets from the specific realm of food quality. The latter includes both large-scale, diverse binary-class food datasets and few-shot, multi-class food datasets. Across all these scenarios, AdaptCL consistently exhibits robust performance, demonstrating its flexibility and general applicability in handling heterogeneous datasets.

## Repository Contents

- **Code**: The code for the AdaptCL method.
- **Datasets**: Relevant datasets used in the experiments.
- **Documentation**: Detailed documentation on how to use the code and reproduce the experiments.
- **Citation**: Please cite our paper if you use this code or find our work helpful.

## Usage

- Detailed instructions on how to use the code and reproduce the experiments are available in the [documentation](documentation/).

## Citation

If you use this code or find our work helpful, please consider citing our paper:

[Paper Title](Link to the Paper)

## License

This project is licensed under the [LICENSE NAME](LICENSE).

