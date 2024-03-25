import numpy as np 
import torch.utils.data as data
import torch

def load_all(train_path, valid_path, test_path):
	""" We load all the three file here to save time in each epoch. """
	train_dict = np.load(train_path, allow_pickle=True).item()
	valid_dict = np.load(valid_path, allow_pickle=True).item()
	test_dict = np.load(test_path, allow_pickle=True).item()

	# get the number of users and items
	user_num, item_num = 0, 0
	user_num = max(user_num, max(train_dict.keys()))
	user_num = max(user_num, max(valid_dict.keys()))
	user_num = max(user_num, max(test_dict.keys()))
	
	train_data, valid_gt, test_gt = [], [], []
	for user, items in train_dict.items():
		item_num = max(item_num, max(items))
		for item in items:
			train_data.append([int(user), int(item)])
	for user, items in valid_dict.items():
		item_num = max(item_num, max(items))
		for item in items:
			valid_gt.append([int(user), int(item)])
	for user, items in test_dict.items():
		item_num = max(item_num, max(items))
		for item in items:
			test_gt.append([int(user), int(item)])

	# print data shape
	print('user_num: %d, item_num: %d' % (user_num, item_num))
	print('train_data: %d, valid_data: %d, test_data: %d' % (len(train_data), len(valid_gt), len(test_gt)))
	
	return user_num+1, item_num+1, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt

class MFData(data.Dataset):
	def __init__(self, features, num_item, train_dict=None, is_training=None):
		super(MFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		self.num_item = num_item
		self.train_dict = train_dict
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]

	# handle negative sampling
	def ng_sample(self):
		assert self.is_training, 'no need to sample when testing'

		self.features_ng = []
		for x in self.features_ps:
			u = x[0]
			j = np.random.randint(self.num_item)
			while j in self.train_dict[u]:
				j = np.random.randint(self.num_item)
			self.features_ng.append([u, j])

		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]

		self.features_fill = self.features_ps + self.features_ng
		self.labels_fill = labels_ps + labels_ng

	def __len__(self):
		return (1 + 1) * len(self.labels)

	def __getitem__(self, idx): # Important function for pytorch, check this out
		features = self.features_fill if self.is_training else self.features_ps
		labels = self.labels_fill if self.is_training else self.labels

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]

		return user, item, label

class CBFData(data.Dataset):
    def __init__(self, user_item_pairs, num_items, category_features, visual_features, user_profiles, train_dict=None, is_training=True):
        super(CBFData, self).__init__()
        self.user_item_pairs = user_item_pairs
        self.num_items = num_items
        self.category_features = category_features
        self.visual_features = visual_features
        self.user_profiles = user_profiles
        self.train_dict = train_dict
        self.is_training = is_training
        self.labels = [1] * len(user_item_pairs)  
        
        if self.is_training:
            self.ng_sample()

    def ng_sample(self):
        self.user_item_pairs_ng = []
        for user_id, _ in self.user_item_pairs:
            neg_item = np.random.randint(self.num_items)
            while neg_item in self.train_dict.get(user_id, []):
                neg_item = np.random.randint(self.num_items)
            self.user_item_pairs_ng.append((user_id, neg_item))
        
        self.features_fill = self.user_item_pairs + self.user_item_pairs_ng
        self.labels_fill = [1] * len(self.user_item_pairs) + [0] * len(self.user_item_pairs_ng)  # 1 for positive, 0 for negative

    def __len__(self):
        return len(self.features_fill)

    def __getitem__(self, idx):
        user_id, item_id = self.features_fill[idx]
        label = self.labels_fill[idx]
        
        # Fetch user profile features
        user_profile = self.user_profiles[user_id]
        user_category_feature = user_profile['category_sum']
        user_visual_feature = user_profile['visual_sum']
        
        # Fetch and prepare item category and visual features
        category_feature = self.category_features[item_id]
        visual_feature = self.visual_features[item_id]
        
        user_category_feature_tensor = torch.tensor(user_category_feature, dtype=torch.float32)
        user_visual_feature_tensor = torch.tensor(user_visual_feature, dtype=torch.float32)
        category_feature_tensor = torch.tensor(category_feature, dtype=torch.float32)
        visual_feature_tensor = torch.tensor(visual_feature, dtype=torch.float32)
        
        return user_category_feature_tensor, user_visual_feature_tensor, category_feature_tensor, visual_feature_tensor, torch.tensor(label, dtype=torch.float32)
