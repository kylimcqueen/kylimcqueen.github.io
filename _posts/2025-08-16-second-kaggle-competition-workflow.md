---
layout: post
title: A workflow for my second Kaggle competition
subtitle: My plan to stay on track 
tags: [kaggle]
---

# A workflow for my second Kaggle competition

*August 15, 2025*

Last week I completed the Advanced House Prices practice Kaggle competition. My next challenge is the 
[NeurIPS - Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025). 

**Objective**: Use text data about polymer structure to predict five metrics that determine performance of a given polymer

**Goal**: Place in the top half of the leaderboard

I looked through the tutorial notebook posted by the competition hosts and then used Claude to create a workflow 
to keep me on track as I progress through the competition. I expect it will take me some time. As I progress 
through the competition, I’ll post more about what I’ve learned at each major step. 
Check back soon to see where I’m at. See the workflow below:

## Open Polymer Prediction Competition Workflow

### Understanding the Problem

#### What does the data look like? 

The input consists of **SMILES** structures. SMILES stands for Simplified Molecular Line Input System. 
If you studied a science-related subject in college like I did, you may have taken organic chemistry. 
And if you took organic chemistry, you may have been exposed to the IUPAC system for chemical nomenclature. 
IUPAC is a standardized set of rules for naming chemical structures. 
SMILES has the same idea, except the goal of SMILES is to _represent molecular structures in a computer-readable format_.

#### What's the evaluation metric?

Weighted Mean Absolute Error (wMAE)

## Baseline Implementation

* Run the tutorial notebook
* Understand each model's approach
* Document baseline scores

## Systematic Parameter Tuning

* Learning rates, LSTM units, layers, dropout
* Use proper validation (not just training score)
* Track what works and what doesn't

## Model Architecture Exploration

* Different neural networks (GRU, Transformer, CNN for sequences)
* Traditional ML approaches
* Hybrid approaches

## Advanced Techniques

* Feature engineering (polymer-specific features)
* Ensemble methods
* Cross-validation strategies



