# Glucose Level Prediction

## Overview
This repository provides an easy-to-run pipeline for predicting blood glucose levels using machine learning techniques. The model explores relationships between **Alpha-Amylase** and **Electric Resistance** to estimate glucose levels accurately.

## Features
- Machine learning-based prediction of blood glucose levels
- Simple and efficient pipeline implementation
- Command-line execution for ease of use

## Installation
Before running the pipeline, ensure you have Python installed along with the necessary dependencies. You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage
To execute the prediction pipeline, run the following command in your terminal:

```bash
python pipeline.py
```

## Dependencies
Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- `seaborn`

## Project Structure
```
📂 Glucose_Level_Prediction
├── data/          # Data processing scripts
│   ├── __pycache__
│   ├── array_torch.py
│   ├── load_data.py
│   ├── visualization.py
├── models/        # Model-related scripts
│   ├── __pycache__
│   ├── metrics.py
│   ├── second_model.py
│   ├── train.py
├── best_model.pth # Trained model weights
├── pipeline.py    # Main script to run the pipeline
├── Real_Cleaned_Data.xlsx # Processed dataset
├── README.md      # Project documentation
└── requirements.txt  # Required dependencies
```

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or suggestions, feel free to reach out via GitHub issues.

