import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=32, dropout=0, mean=0):
        super(MF, self).__init__()
        self.model_name = 'MF'
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

class ContentBasedModel(nn.Module):
    def __init__(self, num_categories, num_visual_features, hidden_dim):
        super(ContentBasedModel, self).__init__()
        self.model_name = 'CBF'
        self.user_category_fc = nn.Linear(num_categories, hidden_dim) # User category pathway
        self.user_visual_fc = nn.Linear(num_visual_features, hidden_dim) # User visual pathway
        self.item_category_fc = nn.Linear(num_categories, hidden_dim) # Item category pathway
        self.item_visual_fc = nn.Linear(num_visual_features, hidden_dim) # Item visual pathway
        self.combined_fc = nn.Linear(hidden_dim * 4, hidden_dim) # Combined features for prediction
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, user_category, user_visual, item_category, item_visual):
        # Process features through respective pathways
        user_category_out = F.relu(self.user_category_fc(user_category))
        item_category_out = F.relu(self.item_category_fc(item_category))
        user_visual_out = F.relu(self.user_visual_fc(user_visual))
        item_visual_out = F.relu(self.item_visual_fc(item_visual))
        
        # Combine all pathways
        combined_features = torch.cat((user_category_out, item_category_out, user_visual_out, item_visual_out), dim=1)
        
        # Further processing for final prediction
        combined_out = F.relu(self.combined_fc(combined_features))
        output = torch.sigmoid(self.output_layer(combined_out))
        return output

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=32, dropout=0.2):
        super(NCF, self).__init__()
        self.model_name = 'NCF'
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_size)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_size)

        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size)

        # MLP layers
        MLP_modules = []
        input_size = embedding_size * 2  # user + item embeddings
        layers=[int(64), int(32), int(16)]
        for layer_size in layers:
            MLP_modules.append(nn.Linear(input_size, layer_size))
            MLP_modules.append(nn.ReLU())
            MLP_modules.append(nn.Dropout(p=dropout))
            input_size = layer_size
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # Final layer
        self.output_layer = nn.Linear(embedding_size + layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.output_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user_indices, item_indices):
        # GMF part
        user_embedding_gmf = self.user_embedding_gmf(user_indices)
        item_embedding_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_embedding_gmf * item_embedding_gmf

        # MLP part
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat((user_embedding_mlp, item_embedding_mlp), -1)
        mlp_output = self.MLP_layers(mlp_input)

        # Concatenate GMF and MLP parts
        concat = torch.cat((gmf_output, mlp_output), -1)

        # Final prediction
        prediction = torch.sigmoid(self.output_layer(concat))
        return prediction.squeeze()
    

