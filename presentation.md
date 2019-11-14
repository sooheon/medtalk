footer: Factor AI
slidenumbers: true

# Hands on Medical Applications of Deep Learning

**2019/11/7**

Sooheon Kim
Paul Warren

---

# Table of Contents

- img semantic segmentation
- federated train
- nlp + active learning
- classification/reg (patient outcomes, kaggle)
- industry trends

Be sure to talk about common mistakes, where things go wrong

---

## Machine Learning Process

Descriptive, Predictive, Prescriptive

---

## AI as applied to medicine
- Narrow applications
  - Diagnosis, Prognosis, Treatment
- Operations
  - Hospital flow, nursing staffing optimization
- Relevance
  - Adapting to situation, provider and patient, surfacing *relevant* information.

---

## Narrow: Sepsis prediction (HCA)

- #1 most expensive condition in hospital, $24bn annually (US)[^1]
- 270,000 deaths per annum[^1]
  - 1h delay in treatment ≈ 4 ~ 7% increased mortality
- SPOT (Sepsis Prevention and Optimization of Therapy)
  - Real time EHR monitoring (vitals, lab results, nursing reports)
  - Alert staff for sepsis
- Reduce sepsis mortality by 23% (2017 ~ 2018)[^2]


[^1]: Statistical Brief #204. Healthcare Cost and Utilization Project (HCUP). April 2016. Agency for Healthcare Research and Quality, Rockville, MD.

[^2]: https://www.modernhealthcare.com/care-delivery/hca-uses-predictive-analytics-spot-sepsis-early

---

## Operations: Dirty Hospital Bed Detection

- Computer vision on CCTV cameras to detect *used*, *empty* hospital beds
- Alert situation room to presense for nurse dispatch
- PoC deployed in one hospital in Germany, phase 2 is 19 more hospitals
  - Success is 1 or 2 more surgeries scheduled per day

Currently under development in Germany

---

## Tooling: Nvidia Clara

Imaging & Genomics

---

## Challenge: Data 

Data is lifeblood of AI

Bottlenecks to data: gathering, cleaning, labeling

Gathering is gated by access (patient records)

Cleaning and labeling is gated by labor (data engineers, domain experts)

---

## Tooling: TensorFlow Federated

Collaborate on shared model in cloud, without centralizing data.

---

## Model Training and Use: Client / Server


---

## Model Training and Use: Federated

![inline](federated_training.png)
