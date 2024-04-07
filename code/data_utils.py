import os
import numpy as np 
import torch
import torch.utils.data as data
from sklearn.preprocessing import OneHotEncoder

# Data for MF model

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
	
# Data for CBF model

def load_data():

	# Load data
	train_dict = np.load('../data/training_dict.npy', allow_pickle=True).item()
	valid_dict = np.load('../data/validation_dict.npy', allow_pickle=True).item()
	test_dict = np.load('../data/testing_dict.npy', allow_pickle=True).item()
	category_features = np.load(os.path.join('../data/category_feature.npy'), allow_pickle=True).item()
	visual_features = np.load(os.path.join('../data/visual_feature.npy'), allow_pickle=True).item()

	# Get the number of users and items
	user_num = max(max(train_dict), max(valid_dict, default=-1), max(test_dict, default=-1)) + 1

	item_num = max(
		max((max(items, default=-1) for items in train_dict.values()), default=-1),
		max((max(items, default=-1) for items in valid_dict.values()), default=-1),
		max((max(items, default=-1) for items in test_dict.values()), default=-1)
	) + 1

	print('Number of users: %d, Number of items: %d' % (user_num, item_num))

	# Prepare training, validation, and test data
	train_data = [[int(user), int(item)] for user, items in train_dict.items() for item in items]
	valid_gt = [[int(user), int(item)] for user, items in valid_dict.items() for item in items]
	test_gt = [[int(user), int(item)] for user, items in test_dict.items() for item in items]
	print('Training samples: %d, Validation samples: %d, Test samples: %d' % (len(train_data), len(valid_gt), len(test_gt)))

	# Load item features
	category_feature_size = len(category_features)
	unique_categories = set(category_features.values())
	category_encoder = OneHotEncoder()
	category_encoder.fit(np.array(list(unique_categories)).reshape(-1, 1))
	category_features_onehot = category_encoder.transform(np.array(list(category_features.values())).reshape(-1, 1)).toarray()
	print('Category features shape: %s' % str(category_features_onehot.shape))

	visual_feature_size = len(visual_features)
	example_key = next(iter(visual_features.keys()))
	print('Visual features shape: %s' % str(visual_features[example_key].shape))

	# Create user profiles
	def create_user_profiles(interaction_dict, category_features, visual_features):

		user_profiles = {
			user_id: {
				'category_sum': np.zeros(category_features_onehot.shape[1]), 
				'visual_sum': np.zeros(visual_features[0].shape),
				'count': 0
			}
			for user_id in interaction_dict
		}

		for user_id, items in interaction_dict.items():
			for item_id in items:
				user_profiles[user_id]['category_sum'] += category_features[item_id]
				user_profiles[user_id]['visual_sum'] += visual_features[item_id]
				user_profiles[user_id]['count'] += 1

		# Averaging the features for each user profile
		for profile in user_profiles.values():
			if profile['count'] > 0:
				profile['category_sum'] /= profile['count']
				profile['visual_sum'] /= profile['count']

		return user_profiles

	train_user_profiles = create_user_profiles(train_dict, category_features_onehot, visual_features)
	valid_user_profiles = create_user_profiles(valid_dict, category_features_onehot, visual_features)
	test_user_profiles = create_user_profiles(test_dict, category_features_onehot, visual_features)

	return user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt, category_features, category_features_onehot, visual_features, train_user_profiles, valid_user_profiles, test_user_profiles

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
