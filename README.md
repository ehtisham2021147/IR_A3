# MovieLens Implicit Feedback Recommender  
**Information Retrieval – Assignment 3**  
**Author:** Ehtisham Khalid (2021147)  
**Dataset:** MovieLens-1M

---

## Project Overview

This project builds an **implicit-feedback recommender system** using collaborative filtering techniques. Specifically, it uses **user-movie interactions** from the MovieLens-1M dataset and applies **matrix factorization** using PyTorch. The goal is to recommend movies based on user behavior (positive ratings) without relying on explicit ratings.

---

## Features

- Downloads the **MovieLens-1M dataset** using the Kaggle API
- Converts **explicit ratings to implicit feedback** (ratings ≥ 4 → interaction = 1)
- Applies **negative sampling** to generate training labels
- Implements a **PyTorch-based matrix factorization model**
- Trains using **binary cross-entropy loss**
- Evaluates performance using **Hit Rate @K** and **NDCG @K**

---

## Installation

Before running the notebook, make sure the following dependencies are installed:

```bash
pip install kaggle
pip install pandas numpy scikit-learn torch tqdm
Additionally, configure your Kaggle API credentials:

Visit https://www.kaggle.com → Account → Create API Token.

Download the kaggle.json file.

Upload it in the notebook or place it in the appropriate directory (~/.kaggle/kaggle.json).

Project Structure
Component	Description
ratings.csv	Cleaned dataset with ratings ≥ 4 marked as interactions
negative_sampling()	Adds user-item pairs with no interaction
InteractionDataset	Custom PyTorch dataset for training
MatrixFactorizationModel	PyTorch model with user/item embeddings
train()	Training loop using BCE loss
evaluate()	Calculates Hit@K and NDCG@K for test users

Dataset
This project uses the MovieLens-1M dataset, which contains:

1 million+ ratings

6,000+ users

3,000+ movies

Ratings are converted to binary interactions:

rating >= 4.0 → interaction = 1

Negative samples are generated for training

Model
A basic matrix factorization model is implemented using PyTorch:

python
Copy
Edit
class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

    def forward(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)
        return (u * i).sum(1)
Training is done using BCEWithLogitsLoss and Adam optimizer.

Evaluation
Two metrics are used to evaluate top-k recommendation performance:

Hit@K – How often is the ground-truth movie in the top-K predicted list?

NDCG@K – Measures ranking quality based on ground-truth relevance.

How to Run
Clone this repository or open the notebook in Colab.

Upload your kaggle.json file.

Run the notebook cells step-by-step:

Download dataset

Preprocess data

Train matrix factorization model

Evaluate recommendations

Sample Output
graphql
Copy
Edit
Hit@10: 0.634
NDCG@10: 0.423
These results may vary based on training epochs, embedding size, and random seeds.

Future Improvements
Add support for content-based filtering using movie metadata

Use autoencoders or neural collaborative filtering

Tune embedding dimensions and learning rates

Implement early stopping and cross-validation

License
This project is created for academic purposes. Please acknowledge the author if you reference or reuse any part of the implementation.
