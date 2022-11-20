# Relation Extraction Task

This is 🚀 Team NLP-09's repository for the 2nd boostcamp AI Tech competition (2022.11.14 ~ 2022.12.01 19:00).

The competition is on sentence-level Relation Extraction, RE.

## 📄 Guideline

### 1. Setup
```bash
make setup
```

### 2. Code formatting & Check liint
```bash
make style
```

### 3. Code testing
```bash
make test
```

### 4. Run Train
```bash
cd src
python train.py
```
You can see more option to train in `src/utils/util`.

### 5. Run Dashboard
```bash
make dashboard
```
Then you can acess dashboard by your web browser.
(If you use **VScode**, check [issue](https://github.com/boostcampaitech4lv23nlp2/level2_klue_nlp-level2-nlp-09/issues/15))

### 6. Run Inference
```bash
cd src
python inference.py
```
You can see more option to inference in `src/utils/util`.

