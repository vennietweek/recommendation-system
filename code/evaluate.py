import numpy as np
import torch
from evaluate_cbf import calculate_diversity, f1_score

def evaluate(model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, device):
    recommends = []
    for i in range(len(top_k)):
        recommends.append([])

    with torch.no_grad():
        pred_list_all = []
        for i in gt_dict.keys():  # for each user
            if len(gt_dict[i]) != 0:  # if
                user = torch.full((item_num,), i, dtype=torch.int64).to(device)  # create n_item users for prediction
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

'''
def evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag):
	recommends = []
	for i in range(len(top_k)):
		recommends.append([])

	with torch.no_grad():
		pred_list_all = []
		for i in gt_dict.keys(): # for each user
			if len(gt_dict[i]) != 0: # if 
				user = torch.full((item_num,), i, dtype=torch.int64).to(args.device) # create n_item users for prediction
				item = torch.arange(0, item_num, dtype=torch.int64).to(args.device) 
				prediction = model(user, item)
				prediction = prediction.detach().cpu().numpy().tolist()
				for j in train_dict[i]: # mask train
					prediction[j] -= float('inf')
				if flag == 1: # mask validation
					if i in valid_dict:
						for j in valid_dict[i]:
							prediction[j] -= float('inf')
				pred_list_all.append(prediction)

		predictions = torch.Tensor(pred_list_all).to(args.device) # shape: (n_user,n_item)
		for idx in range(len(top_k)):
			_, indices = torch.topk(predictions, int(top_k[idx]))
			recommends[idx].extend(indices.tolist())
	return recommends

def metrics(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag):
	RECALL, NDCG = [], []
	recommends = evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag)

	for idx in range(len(top_k)):
		sumForRecall, sumForNDCG, user_length = 0, 0, 0
		k=-1
		for i in gt_dict.keys(): # for each user
			k += 1
			if len(gt_dict[i]) != 0:
				userhit = 0
				dcg = 0
				idcg = 0
				idcgCount = len(gt_dict[i])
				ndcg = 0

				for index, thing in enumerate(recommends[idx][k]):
					if thing in gt_dict[i]:
						userhit += 1
						dcg += 1.0 / (np.log2(index+2))
					if idcgCount > 0:
						idcg += 1.0 / (np.log2(index+2))
						idcgCount -= 1
				if (idcg != 0):
					ndcg += (dcg / idcg)

				sumForRecall += userhit / len(gt_dict[i])
				sumForNDCG += ndcg
				user_length += 1

		RECALL.append(round(sumForRecall/user_length, 4))
		NDCG.append(round(sumForNDCG/user_length, 4))

	return RECALL, NDCG
'''
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
                            
