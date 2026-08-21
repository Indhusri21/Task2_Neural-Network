# Student Performance Classification using Neural Network

A 2-layer Neural Network built completely from scratch using only NumPy to classify students into Pass or Fail categories based on synthetic academic performance metrics.


Task Overview

The objective of this project is to implement the end-to-end learning process of a binary classification Neural Network .


Features & Dataset Specification

The synthetic dataset consists of 200 student records generated with the following features:
 Study Hours: Continuous range (1 to 10 hours)
 Attendance: Continuous percentage (40% to 100%)
 Past Marks: Continuous percentage (40% to 100%)
 Assignment Marks: Continuous percentage (40% to 100%)

Weighted Passing Logic:
Total Score = (0.3 * Study Hours * 10) + (0.3 * Attendance) + (0.2 * Past Marks) + (0.2 * Assignment Marks)

Students with a Total Score > 65 are labeled as Pass (1), otherwise Fail (0).


Model Architecture

 Input Layer: 4 Features (Standardized using Z-score)
 Hidden Layer: 8 Neurons (ReLU Activation)
 Output Layer: 1 Neuron (Sigmoid Activation)


Core Components Implemented

 Feature Standardization: Z = (X - Mean) / Std_Dev for stable numerical convergence.
 Activation Functions:
  - ReLU: max(0, z) and its derivative (z > 0)
  - Sigmoid: 1 / (1 + exp(-z))
 Loss Function: Binary Cross-Entropy Loss with clipping (epsilon = 1e-15) for numerical stability.
 Optimization: Batch Gradient Descent with Backpropagation over 1000 epochs (Learning Rate = 0.1).


Training & Model Performance

 Initial Loss: ~0.6931 (Random initialization baseline)
 Final Loss: ~0.0569
 Final Model Accuracy: 99.00%

Sample Prediction Test:
Input Features: [8 Study Hours, 85% Attendance, 75% Past Marks, 80% Assignment Marks]
Prediction: PASS (Probability: 100.00%)


Project Setup & Usage

Prerequisites:
    >> Python 3.x
    >> numpy
    >> pandas
    >> matplotlib

Installation:
pip install numpy pandas matplotlib

Execution:
python main.py