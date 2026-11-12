# Retrieval-Augmented Sentence Planning for Dialogue Summarization

This repository contains the implementation for the paper **"Retrieval-Augmented Sentence Planning for Dialogue Summarization"**
(Findings of EMNLP 2025).

The released code was written by the authors and is intended to reproduce the
main results reported in the paper.

## Repository layout

- `src/retriever.py` - dialogue-turn retrieval index
- `src/planner.py` - sentence planning model
- `src/train.py` - fine-tuning script

## Environment

```
python>=3.9
torch>=2.1
numpy>=1.24
```

Install the dependencies with `pip install -r requirements.txt`.

## Training

```
python -m src.train --config configs/main.yaml
```

## Citation

```
@inproceedings{retrieval-augmented2025,
  title={Retrieval-Augmented Sentence Planning for Dialogue Summarization},
  author={Scott, Jeffrey and Raman, Priya},
  booktitle={Findings of EMNLP 2025},
  year={2025}
}
}
