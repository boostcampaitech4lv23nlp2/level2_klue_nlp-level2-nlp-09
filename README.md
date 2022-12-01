# Relation Extraction Task

This is 🌏Earth Coding Lab's repository for the 2nd boostcamp AI Tech competition (2022.11.14 ~ 2022.12.01 19:00).

The competition is on sentence-level Relation Extraction.

- Here is our Wrap-Up Report [link]

- Here is our Presentation [link]

## Contributors

|Yong Woo Song|Woo Bin Park|Jin Myeong Ahn|Yeong Cheol Kang|Gang Hyeok Lee|
|:-:|:-:|:-:|:-:|:-:|
|<img src='https://avatars.githubusercontent.com/facerain' height=120 width=120></img>|<img src='https://avatars.githubusercontent.com/wbin0718' height=120 width=120></img>|<img src='https://avatars.githubusercontent.com/jinmyeongAN' height=120 width=120></img>|<img src='https://avatars.githubusercontent.com/kyc3492' height=120 width=120></img>|<img src='https://avatars.githubusercontent.com/ghlrobin' height=120 width=120></img>
[Github](https://github.com/facerain)|[Github](https://github.com/wbin0718)|[Github](https://github.com/jinmyeongAN)|[Github](https://github.com/kyc3492)|[Github](https://github.com/ghlrobin)

`Yong Woo Song` &nbsp; : &nbsp; Result Analysis • Paper Research • Data Augmentation • Model Implementation <br>
`Woo Bin Park` &nbsp; : &nbsp;  Loss Function Analysis • Model Ensemble • Model Implementation <br>
`Jin Myeong Ahn` &nbsp; : &nbsp;  Code Refactoring • Model Implementation <br>
`Yeong Cheol Kang` &nbsp; : &nbsp; MLFlow Customization • Model Customization • Model Implementation <br>
`Gang Hyeok Lee` &nbsp; : &nbsp; Paper Research • Data Cleaning • Hyperparameter Tuning • Model Implementation <br>

## Hardware Used
- NVIDIA TELSA V100
- Ubuntu 18.04

## 📄 Guideline

### 1. Setup
Install all the prerequisites in one go.
```bash
make setup
```

### 2. Code formatting & Check lint
```bash
make style
```

### 3. Code Testing
```bash
make test
```

### 4. Training
```bash
python main.py
```

### 5. Inference
```bash
python main.py --do_train=False --do_inference
```

### 6. Run Dashboard
```bash
make dashboard
```
Then you can acess dashboard through your web browser.
(If you use **VScode**, check [issue](https://github.com/boostcampaitech4lv23nlp2/level2_klue_nlp-level2-nlp-09/issues/15))

### Data
In this competition, KLUE dataset for relation extraction was used. It can be downloaded from the link below:
https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000075/data/dataset.tar.gz

### EDA
EDA can be seen from eda_basics.ipynb file under eda folder.

## Leaderboard
||Micro F1|AUPRC|Rank|
|-|-|-|-|
|Public|74.5240|76.6242|7th|
|Private|74.5271|81.1231|2nd|
