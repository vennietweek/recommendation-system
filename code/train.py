import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from data_utils import *
from models import *
from evaluate_cbf import *
from evaluate import *

def train_cbf(model_name, user_num, item_num, train_data, valid_dict, train_dict, category_features, category_features_onehot, visual_features, user_profiles, num_categories, num_visual_features, hidden_dim, top_k, epochs, batch_size, lr, device, diversity_param=0.5):

    # construct the train datasets & dataloader
    train_dataset = CBFData(user_item_pairs=train_data, num_items=item_num, category_features=category_features_onehot, visual_features=visual_features, user_profiles=user_profiles, train_dict=train_dict, is_training=True)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    if model_name == 'CBF':
        model = ContentBasedModel(num_categories, num_visual_features, hidden_dim)

    model.to(device)

    criterion = nn.BCEWithLogitsLoss() # pointwise loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_recall = 0
    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_loss = 0

        for batch in train_loader:
            user_category, user_visual, item_category, item_visual, labels = batch
            user_category, user_visual = user_category.to(device), user_visual.to(device)
            item_category, item_visual = item_category.to(device), item_visual.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(user_category, user_visual, item_category, item_visual).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Time elapsed: {time.time() - start_time:.2f}s")

        # Evaluation    
        recommends, results = metrics_cbf(model, top_k, train_dict, valid_dict, user_profiles, category_features, category_features_onehot, visual_features, device, diversity_param, is_training=True)

        # Update best F1 score and save model if necessary
        current_best_f1 = np.mean([results[k]['F1'] for k in results.keys()])
        current_best_recall = np.mean([results[k]['Recall'] for k in results.keys()])
        if current_best_recall > best_recall:
            best_recall = current_best_recall
            best_f1 = current_best_f1
            # Save the model checkpoint
            torch.save(model.state_dict(), f'./models/best_model_{model.model_name}.pth')
            print(f"New best model saved with Average Recall: {best_recall}, Average F1: {best_f1}, model path: ./models/best_model_{model.model_name}.pth")
        print('---'*18)

    print("Training completed.")
    print("Best Average Recall: ", best_recall)
    print("Best Average F1 score: ", best_f1)


def train_cf(model_name, user_num, item_num, train_data, valid_dict, train_dict, category_features, emb_size, top_k, epochs, batch_size, lr, device, dropout=0.5):
    # Load data
    train_dataset = MFData(train_data, item_num, train_dict, True)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    if model_name == 'MF':
        model = MF(user_num, item_num, emb_size, dropout)
    elif model_name == 'NCF':
        model = NCF(user_num, item_num, emb_size, dropout)

    model.to(device)
    loss_function = nn.BCEWithLogitsLoss() # pointwise loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_recall = 0
    best_f1 = 0
    total_loss = 0

    for epoch in range(epochs):
        # train
        model.train() # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        # for each batch
        for idx, (user, item, label) in enumerate(train_loader):
            user, item, label = user.to(device), item.to(device), label.float().to(device)
            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Time elapsed: {time.time() - start_time:.2f}s")

        model.eval()
        recommends, results = metrics(model, top_k, train_dict, valid_dict, valid_dict, item_num, 0, category_features, device)

        # Update best F1 score and save model if necessary
        current_best_f1 = np.mean([results[k]['F1'] for k in results.keys()])
        current_best_recall = np.mean([results[k]['Recall'] for k in results.keys()])
        if current_best_recall > best_recall:
            best_recall = current_best_recall
            best_f1 = current_best_f1
            # Save the model checkpoint
            torch.save(model.state_dict(), f'./models/best_model_{model.model_name}.pth')
            print(f"New best model saved with Average Recall: {best_recall}, Average F1: {best_f1}, model path: ./models/best_model_{model.model_name}.pth")
        print('---'*18)

    print("Training completed.")
    print("Best Recall: ", best_recall)
    print("Best F1 score: ", best_f1)