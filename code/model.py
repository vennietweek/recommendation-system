import torch
import torch.nn as nn
import torch.nn.functional as F


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=32, dropout=0, mean=0):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1) # bias term to add to user side
        self.item_emb = nn.Embedding(num_items, embedding_size) # initialise an item embedding for items in the dataset
        self.item_bias = nn.Embedding(num_items, 1)

        # embedding size is the same for user and item datasets in order to compute inner product

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([mean]), False) # global bias which is learnable
        self.dropout = nn.Dropout(dropout) # randomly drops out some inputs during training to improve robustness of model and prevent overfitting

        self.num_user = num_users

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id) # looks up user embedding based on userid
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return self.dropout((U * I).sum(1) + b_u + b_i + self.mean) # inner product of user and item embeddings + bias terms


class NGCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=32, hidden_units=64, dropout=0.1):
        super(NGCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.item_emb = nn.Embedding(num_items, embedding_size)

        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(embedding_size * 2, hidden_units)
        self.output_layer = nn.Linear(hidden_units, 1)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_emb(user_ids)
        item_embedding = self.item_emb(item_ids)

        # Concatenate user and item embeddings
        embeddings = torch.cat([user_embedding, item_embedding], dim=1)
        embeddings = self.dropout(embeddings)

        # Pass through hidden layer
        hidden_output = F.relu(self.hidden_layer(embeddings))
        hidden_output = self.dropout(hidden_output)

        # Output layer
        output = self.output_layer(hidden_output)
        return torch.sigmoid(output.squeeze())  # Apply sigmoid activation function

