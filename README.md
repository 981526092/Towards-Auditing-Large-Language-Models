# Auditing Large Language Models: Tools for Detecting Text-Based Stereotypes

## Overview:
This repository offers comprehensive scripts designed for training stereotype classifiers. These tools are adept at identifying stereotypes across multiple dimensions and at varying levels of text granularity, including both sentence and token levels.

## Resources:

1. **Sentence Level Stereotype Detector**  
   Access here: [Sentence-Level Stereotype Detector](https://huggingface.co/wu981526092/Sentence-Level-Stereotype-Detector)

2. **Token Level Stereotype Detector**  
   Explore at: [Token-Level Stereotype Detector](https://huggingface.co/wu981526092/Token-Level-Stereotype-Detector/settings)

3. **MGSD Dataset**  
   Available at: [MGSD Dataset on Hugging Face](https://huggingface.co/datasets/wu981526092/MGSD)

4. **Stereotype Elicitation Prompt Library**  
   Find here: [Stereotype Elicitation Prompt Library](https://huggingface.co/datasets/wu981526092/Stereotype-Elicitation-Prompt-Library)


## Quick Start
To train all classifiers, execute the following script:

```bash
bash Training-All.sh
```

## Detailed Instructions

### Sentence-Level Training

#### Multi-Dimensional Classifier
To train a multi-dimensional classifier at the sentence level, run:
```bash
bash MD-SL-Training.sh
```

#### Single-Dimensional Classifier
To train a single-dimensional classifier at the sentence level, run:
```bash
bash SD-SL-Training.sh
```

### Token-Level Training

#### Multi-Dimensional Classifier
To train a multi-dimensional classifier at the token level, run:
```bash
bash MD-TL-Training.sh
```

#### Single-Dimensional Classifier
To train a single-dimensional classifier at the token level, run:
```bash
bash SD-TL-Training.sh
```
