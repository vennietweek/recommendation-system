import numpy as np
import torch

def evaluate(model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, device):
    recommends = []
    for i in range(len(top_k)):
        recommends.append([])

    with torch.no_grad():
        pred_list_all = []
        for i in gt_dict.keys():
            if len(gt_dict[i]) != 0: 
                user = torch.full((item_num,), i, dtype=torch.int64).to(device) 
                item = torch.arange(0, item_num, dtype=torch.int64).to(device)
                prediction = model(user, item)
                prediction = prediction.detach().cpu().numpy().tolist()
                for j in train_dict[i]:  # mask train
                    prediction[j] -= float('inf')
                if flag == 1:  # mask validation
                    if i in valid_dict:
                        for j in valid_dict[i]:
                            prediction[j] -= float('inf')
                pred_list_all.append(prediction)

        predictions = torch.Tensor(pred_list_all).to(device)  # shape: (n_user,n_item)
        for idx in range(len(top_k)):
            _, indices = torch.topk(predictions, int(top_k[idx]))
            recommends[idx].extend(indices.tolist())
    return recommends

def metrics(model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, item_categories, device):
    results = {}
    recommends = evaluate(model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, device)

    for idx, k in enumerate(top_k):
        sumForRecall, sumForNDCG, sumForIld, user_length, total_diversity = 0, 0, 0, 0, 0
        k_recalls, k_ndcgs, k_ilds = [], [], []
        
        for user, recs in enumerate(recommends[idx]):
            if user not in gt_dict or not gt_dict[user]:
                continue

            gt_items = set(gt_dict[user])
            hit = sum(item in gt_items for item in recs)
            recall = hit / len(gt_items)
            k_recalls.append(recall)

            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recs) if item in gt_items)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            k_ndcgs.append(ndcg)

            recommended_categories = [item_categories[item] for item in recs]
            ild = calculate_diversity(recommended_categories)
            sumForIld += ild
            k_ilds.append(ild)

            sumForRecall += recall
            sumForNDCG += ndcg
            user_length += 1

        avg_recall = sum(k_recalls) / len(k_recalls) if k_recalls else 0
        avg_ndcg = sum(k_ndcgs) / len(k_ndcgs) if k_ndcgs else 0
        avg_ild = sumForIld / user_length if user_length else 0
        avg_f1 = f1_score(avg_recall, avg_ild)

        results[k] = {'Recall': avg_recall, 'NDCG': avg_ndcg, 'ILD': avg_ild, 'F1': avg_f1}
        print(f"Top-{k}: Avg Recall: {avg_recall:.4f}, Avg NDCG: {avg_ndcg:.4f}, Avg ILD: {avg_ild:.4f}, Avg F1 Score: {avg_f1:.4f}")

    return recommends, results

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Recall: {} NDCG: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]])))
    if test_result is not None: 
        print("[Test]: Recall: {} NDCG: {} ".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]])))

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
    if not np.isnan(ndcg_score) and not np.isnan(ild_score):  
        f1 = 2 * (ndcg_score * ild_score) / (ndcg_score + ild_score)
    else:
        f1 = 0  
    return f1

def rerank_for_diversity(initial_recommendations, item_categories, diversity_param):
    reranked_items = []
    category_penalty = {}  # Dictionary to keep track of category penalties

    for item_id, score in initial_recommendations:
        category = item_categories[item_id]
        penalty = category_penalty.get(category, 0) # Get penalty for this category
        diversity_score = score * (1 - diversity_param) - penalty * diversity_param 
        reranked_items.append((item_id, diversity_score))
        category_penalty[category] = category_penalty.get(category, 0) + 1 # Update category penalty to 1

    # Sort items by diversity_score
    reranked_items.sort(key=lambda x: x[1], reverse=True)
    reranked_item_ids = [item_id for item_id, _ in reranked_items]

    return reranked_item_ids

def evaluate_cbf(model, top_k, train_dict, gt_dict, user_profiles, category_features, category_features_one_hot, visual_features, device, diversity_param, is_training):
    recommends = {k: [] for k in top_k}
    diversity_scores = {k: [] for k in top_k}
    
    with torch.no_grad():
        pred_list_all = []
        for user_id, true_items in gt_dict.items():
            if not true_items:  
                continue
            
            # Get user features
            user_category = torch.tensor(user_profiles[user_id]['category_sum'], dtype=torch.float32).unsqueeze(0)
            user_visual = torch.tensor(user_profiles[user_id]['visual_sum'], dtype=torch.float32).unsqueeze(0)

            # Get items that the user has not interacted with
            interacted_items = train_dict[user_id] if user_id in train_dict else []
            item_ids = [i for i in range(len(category_features_one_hot)) if i not in interacted_items]

            # Remove validation items if not in training

            # Get item features
            item_category_tensor = torch.stack([torch.tensor(category_features_one_hot[i], dtype=torch.float32) for i in item_ids])
            item_visual_tensor = torch.stack([torch.tensor(visual_features[i], dtype=torch.float32) for i in item_ids])

            # Get predictions from the model
            scores = model(user_category.repeat(len(item_ids), 1), user_visual.repeat(len(item_ids), 1), item_category_tensor, item_visual_tensor).squeeze()

            # Generate recommendation pool for reranking
            recommendation_pool_size = max(top_k) * 5
            top_scores, top_indices = torch.topk(scores, recommendation_pool_size)
            recommendation_pool = [(item_ids[i], scores[i].item()) for i in top_indices.cpu().numpy()]

            for k in top_k:
                initial_recommendations = recommendation_pool[:k]
                final_recommendations = []

                if not is_training and diversity_param > 0:
                    rerank_input = [(item_id, score) for item_id, score in initial_recommendations]
                    final_recommendations = rerank_for_diversity(rerank_input, {i: category_features[i] for i, _ in rerank_input}, diversity_param)
                else:
                    final_recommendations = [item_id for item_id, _ in initial_recommendations]

                # Compute ild_score based on final recommendations
                final_categories = [category_features[item_id] for item_id in final_recommendations]
                ild_score = calculate_diversity(final_categories)

                # Store the final recommendations and their diversity score
                recommends[k].append(final_recommendations)
                diversity_scores[k].append(ild_score)

    return recommends, diversity_scores

def metrics_cbf(model, top_k, train_dict, gt_dict, user_profiles, category_features, category_features_one_hot, visual_features, device, diversity_param=0.5, is_training=True):
    
    recommends, diversity_scores = evaluate_cbf(model, top_k, train_dict, gt_dict, user_profiles, category_features, category_features_one_hot, visual_features, device, diversity_param, is_training)

    results = {} 
    
    for k in top_k:
        sumForRecall, sumForNDCG, user_length = 0, 0, 0 # Initialize variables for average calculation
        user_ids_list = list(gt_dict.keys())  # Ensure we have a consistent order
        
        for idx, user_id in enumerate(user_ids_list):
            true_items = gt_dict[user_id]
            if len(true_items) == 0:
                continue  # Skip users with no ground truth data
            
            if idx >= len(recommends[k]):  # Guard against index out of bounds
                continue
            
            recommended_items = recommends[k][idx]  # Access by index
            userhit, dcg, idcg = 0, 0, 0
            idcgCount = len(true_items)
            
            for index, item in enumerate(recommended_items[:k]):
                if item in true_items:
                    userhit += 1
                    dcg += 1.0 / np.log2(index + 2)
                if idcgCount > 0:
                    idcg += 1.0 / np.log2(index + 2)
                    idcgCount -= 1
            
            ndcg = dcg / idcg if idcg != 0 else 0
            recall = userhit / len(true_items) if len(true_items) > 0 else 0
            
            sumForRecall += recall
            sumForNDCG += ndcg
            user_length += 1
        
        avg_recall = sumForRecall / user_length if user_length > 0 else 0
        avg_ndcg = sumForNDCG / user_length if user_length > 0 else 0
        avg_ild = np.mean([diversity_scores[k]]) if k in diversity_scores and diversity_scores[k] else 0
        avg_f1 = np.mean([f1_score(avg_ndcg, avg_ild)])

        results[k] = {'Recall': avg_recall, 'NDCG': avg_ndcg, 'ILD': avg_ild, 'F1': avg_f1}
        print(f"Top-{k}: Avg Recall: {avg_recall:.4f}, Avg NDCG: {avg_ndcg:.4f}, Avg ILD: {avg_ild:.4f}, Avg F1 Score: {avg_f1:.4f}")
    
    return recommends, results

