
# README: Sentiment Analysis with GPT-2 using ColossalAI

This project focuses on fine-tuning a GPT-2 model for sentiment analysis using the hybrid parallel capabilities provided by ColossalAI. Below, you'll find information about the model, dataset, parallel settings, and instructions on how to execute the code.

## Model

I utilize the GPT-2 model for a binary classification task (sentiment analysis). The model is modified to handle the sentiment classification of text inputs, with the output layer tailored to predict two classes (positive and negative).

## Dataset

The dataset used is a CSV file containing text and sentiment labels (binary). The data is split into training and test datasets using a 80-20 ratio.

## Parallel Settings

The experiment leverages ColossalAI's hybrid parallelism, specifically:
- **Pipeline Parallelism:** The model is split into different stages, which can be executed on different devices or nodes, allowing for scalable training.
- **Tensor Parallelism:** The operations within each stage of the pipeline are further parallelized across multiple devices.

This setup is designed to facilitate efficient training over distributed systems, reducing memory usage and accelerating computation.

## Environment Setup

To run this code, ensure you have Python installed with the following packages:
- `torch`
- `transformers`
- `colossalai`
- `pandas`
- `sklearn`

You can install the necessary packages via pip:
```
pip install torch transformers colossalai pandas sklearn
```

## How to Run

1. **Prepare Your Dataset:**
   - Place your training data in a file named `train.csv` in the `data` directory. The CSV file should have at least two columns: `Text` for the input text and `Sentiment` for the binary labels.

2. **Configure the Model:**
   - The model and training parameters can be adjusted in the script. Current settings use GPT-2 with specific hyperparameters like learning rate and batch size in the sample.

3. **Launch the Script:**
   - Run the script using the following command. Ensure that you are in a suitable environment where ColossalAI is configured for distributed training.
   ```
   python your_script_name.py --task sentiment_analysis
   ```

4. **Distributed Training:**
   - The script is set up to initialize a distributed environment automatically using ColossalAI's built-in functions. It handles device allocation and initializes the necessary components for parallel training.

5. **Evaluation:**
   - After training, the model is evaluated on the test set. Performance metrics such as loss and F1 score (if targeted) are outputted.

### Notes

- I don't have access to a Linux system, so I'm unable to execute the codes myself. The hyperparameters and evaluation setup are directly referenced from the original example.
