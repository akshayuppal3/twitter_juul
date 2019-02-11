from authentication import Authenticate
import pandas as pd
import logging
import tweepy
from tqdm import tqdm
import networkx as nx
import argparse
import pickle
import util
import os

logging.basicConfig(level=logging.INFO, format= util.format, filename= os.path.join(util.logdir,"followingData.log"))


class Cascade():

	def __init__(self):
		self.api_list = self.get_api()


	def get_api(self):
		ob = Authenticate()
		api_list = ob.api
		return (api_list)

	# return the dataframe @type= <following,follower>
	def find_connections(self,apis, user_list, typef='following'):
		df = pd.DataFrame()
		print("finding connection for " + typef + " network might take some time\n")
		for user in tqdm(user_list):
			apis.rotate(1)
			api = apis[0]
			try:
				if typef == 'following':
					following = (api.followers_ids(user))  # return list of followers
					data = {
						'userID': user,
						'following_list': following
					}
				elif typef == 'followers':
					friends = (api.friends_ids(user))  # return list of following
					data = {
						'userID': user,
						'followers_list': friends
					}
				else:
					print("wrong type specified")
				df = df.append(data, ignore_index=True)
			except tweepy.TweepError as e:
				continue
		return df

	# now will check the relationship of all node with the source node, does it have folower or following rel.
	def create_cascade_lvl_1(self,source_node, user_list):
		G = nx.DiGraph()  # will add edges directly
		# users type(int)
		first_nodes = list()
		for user in tqdm(user_list):
			apis = self.api_list
			apis.rotate(-1)
			api = apis[0]
			if (user != source_node):
				try:
					relation_obj = api.show_friendship(source_id=(source_node), target_id=(user))[0]
					if ((relation_obj.following == True) or (relation_obj.followed_by == True)):
						if user not in first_nodes:
							first_nodes.append(user)
						if (relation_obj.following == True):
							G.add_edge(source_node, user)
						if (relation_obj.followed_by == True):
							G.add_edge(user, source_node)
				except tweepy.TweepError as e:
					continue
		return (G, first_nodes)

	def create_cascade(self,G, sourceID, user_list, level):
		if len(G) is 0:
			print("getting lvl 1")
			G, first_nodes = self.create_cascade_lvl_1(sourceID, user_list)
			rem_users = list(set(user_list) - set(first_nodes))
			print(rem_users)
			return self.create_cascade(G, first_nodes, rem_users, 1)
		else:
			print("at level ", level + 1)
			print(user_list)
			if (len(user_list) != 0):
				second_user = list()
				apis = self.api_list
				apis.rotate(-1)
				apis = apis[0]
				# find both the following and follower relationship
				df_following = self.find_connections(apis, sourceID, 'following')
				df_followers = self.find_connections(apis, sourceID, 'followers')
				for node in tqdm(sourceID):
					for user in user_list:
						followers = (df_following.following_list[df_following.userID == node].values[0])
						following = (df_followers.followers_list[df_followers.userID == node].values[0])
						if ((user in set(followers)) or (user in set(following))):
							if (user not in second_user):
								second_user.append(user)
							if (user in set(followers)):
								G.add_edge(user, node)
							if user in set(following):
								G.add_edge(node, user)
				path = 'graph_temp' + str(level + 1) + '.pkl'
				with open(path, 'wb') as handle:
					pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)
				rem_users = list(set(user_list) - set(second_user))
				return self.create_cascade(G, second_user, rem_users, level + 1)


if __name__ == '__main__':
	ob = Cascade()
	parser = argparse.ArgumentParser(description='Extracting data from userDataFile')
	parser.add_argument('-i', '--inputFile', help='Specify the input file path for extracting friends', required=False)
	args = vars(parser.parse_args())

