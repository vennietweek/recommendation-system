import torch
from evaluate_cbf import *

def evaluate_ensemble(mf_model, cbf_model, alpha, top_k, train_dict, gt_dict, user_profiles, category_features, category_features_one_hot, visual_features, device, diversity_param, is_training):
    # Create a container for the ensemble's recommendations
    ensemble_recommends = {k: [] for k in top_k}
    
    # Make sure both models are in evaluation mode
    mf_model.eval()
    cbf_model.eval()

    with torch.no_grad():
        for user_id, true_items in gt_dict.items():
            if not true_items:
                continue
            
            # Get the features for CBF model
            user_category = torch.tensor(user_profiles[user_id]['category_sum'], dtype=torch.float32).unsqueeze(0).to(device)
            user_visual = torch.tensor(user_profiles[user_id]['visual_sum'], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get the user and item ids for MF model
            interacted_items = set(train_dict.get(user_id, []))
            item_ids = [item_id for item_id in range(len(category_features_one_hot)) if item_id not in interacted_items]
            
            # Get the predictions from both models
            mf_scores = mf_model(torch.tensor([user_id]*len(item_ids), dtype=torch.long, device=device), torch.tensor(item_ids, dtype=torch.long, device=device))
            cbf_scores = cbf_model(user_category.repeat(len(item_ids), 1), user_visual.repeat(len(item_ids), 1), torch.stack([torch.tensor(category_features_one_hot[i], dtype=torch.float32).to(device) for i in item_ids]), torch.stack([torch.tensor(visual_features[i], dtype=torch.float32).to(device) for i in item_ids]))
            
            # Combine the predictions
            combined_scores = (1 - alpha) * mf_scores + alpha * cbf_scores
            
            # Convert combined scores to a list and exclude already interacted items
            combined_scores = combined_scores.squeeze().tolist()
            for item_id in interacted_items:
                combined_scores[item_id] = -float('inf')

            # Get top-k recommendations
            _, top_indices = torch.topk(torch.tensor(combined_scores), max(top_k))
            recommendation_pool = [item_ids[i] for i in top_indices.cpu().numpy()]

            # Fill in the ensemble_recommends the same way you would in your CBF evaluation
            for k in top_k:
                initial_recommendations = recommendation_pool[:k]
                final_recommendations = rerank_for_diversity(initial_recommendations, category_features, diversity_param) if not is_training and diversity_param > 0 else initial_recommendations
                
                ensemble_recommends[k].append(final_recommendations)

    return ensemble_recommends


def metrics_ensemble(ensemble_recommends, gt_dict, category_features, top_k):
    results = {}
    for k in top_k:
        sum_recall, sum_ndcg, sum_ild, user_count = 0, 0, 0, 0
        for user_id, recommendations in ensemble_recommends[k].items():
            true_items = set(gt_dict.get(user_id, []))
            if not true_items:
                continue
            
            # Calculate recall and ndcg
            hits = [1 if item in true_items else 0 for item in recommendations]
            recall = sum(hits) / len(true_items)
            ndcg = sum(hit / np.log2(index + 2) for index, hit in enumerate(hits) if hit) / sum(1 / np.log2(index + 2) for index in range(min(len(true_items), k)))
            
            # Calculate ILD
            recommended_categories = [category_features[item] for item in recommendations]
            ild = calculate_diversity(recommended_categories)

            sum_recall += recall
            sum_ndcg += ndcg
            sum_ild += ild
            user_count += 1

        avg_recall = sum_recall / user_count if user_count else 0
        avg_ndcg = sum_ndcg / user_count if user_count else 0
        avg_ild = sum_ild / user_count if user_count else 0
        avg_f1 = f1_score(avg_ndcg, avg_ild)
        
        results[k] = {
            'Recall': avg_recall,
            'NDCG': avg_ndcg,
            'ILD': avg_ild,
            'F1': avg_f1
        }
        print(f"Top-{k}: Avg Recall: {avg_recall:.4f}, Avg NDCG: {avg_ndcg:.4f}, Avg ILD: {avg_ild:.4f}, Avg F1 Score: {avg_f1:.4f}")
    
    return results