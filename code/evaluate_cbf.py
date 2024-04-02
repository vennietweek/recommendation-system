import numpy as np
import torch

def calculate_recall(recommended_items, true_items):
    # Convert true_items to set for efficient lookup
    true_item_set = set(true_items)
    # Count the number of recommended items that are in the true_items
    hit_count = sum(item in true_item_set for item in recommended_items)
    # Calculate recall
    recall = hit_count / len(true_item_set) if true_item_set else 0
    return recall

def calculate_ndcg(recommended_items, true_items, k):
    # Convert true_items to set for efficient lookup
    true_item_set = set(true_items)
    # Initialize DCG and IDCG
    dcg = 0.0
    idcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in true_item_set:
            dcg += 1 / np.log2(i + 2)  # +2 because we start counting from position 1
    for i in range(min(k, len(true_item_set))):
        idcg += 1 / np.log2(i + 2)
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

def calculate_diversity(recommended_categories):
    K = len(recommended_categories)
    diversity_score = 0
    
    for i in range(K):
        for j in range(i+1, K):
            if recommended_categories[i] != recommended_categories[j]:
                diversity_score += 1

    # Normalize by the number of possible pairs
    diversity_score /= (K * (K - 1) / 2.0)
    return diversity_score

# Function to calculate the F1 score for diversity and relevance
def f1_score(ndcg_score, ild_score):
    return 2 * (ndcg_score * ild_score) / (ndcg_score + ild_score) if (ndcg_score + ild_score) > 0 else 0

def rerank_for_diversity(initial_recommendations, item_categories, diversity_param=0.5):
    """
    Re-rank recommendations to promote diversity.

    Parameters:
    - initial_recommendations: list of tuples (item_id, score) sorted by score in descending order.
    - item_categories: dictionary mapping item_id to its category.
    - diversity_param: parameter to balance between original score and diversity. Value between 0 and 1.
                       0 means no diversity (original ranking), 1 means only diversity.
    
    Returns:
    - list of item_ids after re-ranking to promote diversity.
    """
    reranked_items = []
    category_penalty = {}  # Dictionary to keep track of category penalties

    for item_id, score in initial_recommendations:
        category = item_categories[item_id]
        penalty = category_penalty.get(category, 0)
        diversity_score = score * (1 - diversity_param) - penalty * diversity_param
        reranked_items.append((item_id, diversity_score))
        category_penalty[category] = category_penalty.get(category, 0) + 1 # Update category penalties

    # Sort items by diversity_score
    reranked_items.sort(key=lambda x: x[1], reverse=True)
    
    # Return the reranked list of item IDs
    return [item_id for item_id, _ in reranked_items]

def evaluate(model, top_k, train_dict, gt_dict, user_profiles, category_features, category_features_one_hot, visual_features, device, diversity_param, is_training):
    recommends = {k: [] for k in top_k}
    diversity_scores = {k: [] for k in top_k}
    
    with torch.no_grad():
        for user_id, true_items in gt_dict.items():
            if not true_items:  # Skip users with no ground truth data
                continue
            
            # Prepare user features
            user_category = torch.tensor(user_profiles[user_id]['category_sum'], dtype=torch.float32).unsqueeze(0).to(device)
            user_visual = torch.tensor(user_profiles[user_id]['visual_sum'], dtype=torch.float32).unsqueeze(0).to(device)

            # Dynamically select item features for items the user has interacted with
            interacted_items = train_dict[user_id] if user_id in train_dict else []
            item_ids = [i for i in range(len(category_features_one_hot)) if i not in interacted_items]

            # Prepare tensors for item features dynamically selected
            item_category_tensor = torch.stack([torch.tensor(category_features_one_hot[i], dtype=torch.float32) for i in item_ids]).to(device)
            item_visual_tensor = torch.stack([torch.tensor(visual_features[i], dtype=torch.float32) for i in item_ids]).to(device)

            # Compute scores for all selected items in one pass
            scores = model(user_category.repeat(len(item_ids), 1), user_visual.repeat(len(item_ids), 1), item_category_tensor, item_visual_tensor).squeeze()

            # Process scores to identify top-k items
            top_scores, top_indices = torch.topk(scores, max(top_k))
            for k in top_k:
                # Get top-k indices and scores
                top_scores, top_indices = torch.topk(scores, k)
                top_k_indices = top_indices.cpu().numpy()
                recommended_items = [item_ids[i] for i in top_k_indices]  # Map back to original item IDs
                
                recommended_categories = [category_features[i] for i in recommended_items]
                ild_score = calculate_diversity(recommended_categories)
                
                # Apply reranking based on diversity if not in training phase
                if not is_training:
                    # Prepare (item ID, score) tuples for reranking
                    rerank_input = [(item_ids[i], scores[top_indices[i]].item()) for i in range(len(top_k_indices))]
                    reranked_indices = rerank_for_diversity(rerank_input, {i: category_features[i] for i in recommended_items}, diversity_param)
                    # Extract reranked item IDs after diversity adjustment
                    reranked_items = [idx for idx, _ in reranked_indices]
                    reranked_categories = [category_features[i] for i in reranked_items]
                    ild_score = calculate_diversity(reranked_categories)
                    # Update recommends with reranked items
                    recommends[k].append(reranked_items)
                else:
                    # No reranking; use original recommendations
                    recommends[k].append(recommended_items)

                diversity_scores[k].append(ild_score)

    return recommends, diversity_scores

def metrics(model, top_k, train_dict, gt_dict, user_profiles, category_features, category_features_one_hot, visual_features, device, diversity_param=0.5, is_training=True):
    recommends, diversity_scores = evaluate(model, top_k, train_dict, gt_dict, user_profiles, category_features, category_features_one_hot, visual_features, device, diversity_param, is_training)
    f1_scores = {k: [] for k in top_k}
    
    for k in top_k:
        recall_scores = []
        ndcg_scores = []
        
        for user_id, recommended_items in zip(gt_dict.keys(), recommends[k]):
            true_items = gt_dict[user_id]
            
            recall = calculate_recall(recommended_items, true_items)
            ndcg = calculate_ndcg(recommended_items, true_items, k)
            ild_score = diversity_scores[k][gt_dict.keys().index(user_id)]  # Fetching the ILD score for the current user
            f1 = f1_score(ndcg, ild_score)
            
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            f1_scores[k].append(f1)

        avg_recall = np.mean(recall_scores)
        avg_ndcg = np.mean(ndcg_scores)
        avg_ild = np.mean(diversity_scores[k])
        avg_f1 = np.mean(f1_scores[k])
        
        print(f"Top-{k}: Avg Recall: {avg_recall:.4f}, Avg NDCG: {avg_ndcg:.4f}, Avg ILD: {avg_ild:.4f}, Avg F1 Score: {avg_f1:.4f}")

        return recommends, avg_f1, avg_ndcg, avg_ild, avg_recall, f1_scores, ndcg_scores, diversity_scores, recall_scores