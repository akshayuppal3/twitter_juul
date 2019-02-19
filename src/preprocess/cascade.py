from authentication import Authenticate
import pandas as pd
import logging
import tweepy
from tqdm import tqdm
import networkx as nx
import git
import argparse
import pickle
import util
from collections import deque
import os

logging.basicConfig(level=logging.INFO, format= util.format, filename= os.path.join(util.logdir,"followingData.log"))


class Cascade():

	def __init__(self):
		self.api_list = self.get_api()

	def get_git_root(self,path):
		git_repo = git.Repo(path, search_parent_directories=True)
		git_root = git_repo.git.rev_parse("--show-toplevel")
		return git_root

	def get_api(self):
		ob = Authenticate()
		api_list = ob.api
		return (api_list)

	# return the dataframe @type= <following,follower>
	def find_connections(self, user_list, typef='following'):
		df = pd.DataFrame()
		print("finding connection for " + typef + " network might take some time\n")
		for user in tqdm(user_list):
			apis = deque(self.api_list)
			apis.rotate(-1)
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

	# @param: user_list= list of users ,df: (dataframe) containing user information
	# @ return a G with node attributes (# friends, # followers, # level)
	def get_node_attributes(self,G,user_list,df,level,source_node=None):
		attr = dict()
		for user in user_list:
			if user in list(df.userID):
				user_data = df.loc[df.userID == user].head(1)
				attr[(user)] = {'friends': list(user_data['friendsCount'])[0],
				                'followers': list(user_data['userFollowersCount'])[0],
				                'level': level}
		nx.set_node_attributes(G,attr)
		if (source_node != None):
			a = df.loc[df.userID == source_node].head(1)
			attr_source = {source_node : {'level': 0,
										 'friends' : list(a['friendsCount'])[0],
										 'followers' : list(a['friendsCount'])[0]}}
			nx.set_node_attributes(G,attr_source)
		return G

	# @param source node, user_list and dataframe
	# now will check the relationship of all node with the source node, does it have folower or following rel.
	def create_cascade_lvl_1(self,source_node, user_list,):
		G = nx.DiGraph()  # will add edges directly
		# users type(int)
		first_nodes = list()
		for user in tqdm(user_list):
			apis = deque(self.api_list)
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

	# get the next level of cascades...
	def create_cascade(self,G, source_id, user_list):
		if (len(user_list) != 0):
			second_user = list()
			# find both the following and follower relationship
			df_following = self.find_connections( source_id, 'following')
			df_followers = self.find_connections( source_id, 'followers')
			for node in tqdm(source_id):
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
			rem_users = list(set(user_list) - set(second_user))
			return (G, second_user, rem_users)

	# @deprecated	# @param get the users for tattoo cascade
	def get_users_tatto_cascade(self, df):
		cascade1 = (df.loc[df['retweetCount'] == 781])  ## need to change this as an argument
		user_tattoos = list(cascade1.userID)
		# getting the first node the sorted list
		tattoo_node = cascade1.head(1)['userID'].values[0]
		user_tattoos.remove(tattoo_node)
		return (tattoo_node, user_tattoos)

if __name__ == '__main__':
	ob = Cascade()
	parser = argparse.ArgumentParser(description='Extracting data from userDataFile')
	parser.add_argument('-i', '--inputFilepath', help='Specify the input file path for extracting friends', required=False)
	args = vars(parser.parse_args())
	if (args['inputFile']):
		logging.info('[NEW] ---------------------------------------------')
		input_path = args['inputFilepath']
		df = pd.read_csv(input_path, lineterminator="\n")
		tattoo_node,user_list = ob.get_users_tatto_cascade(df)
		G, first_level = ob.create_cascade_lvl_1(tattoo_node, user_list)
		rem_users = list(set(user_list) - set(first_level))
		## get level 2
		G,_second_user,rem_users_ = ob.create_cascade(G, first_level, rem_users, 2)
		nx.write_gpickle(G, "/Users/akshayuppal/Desktop/thesis/twitter_juul/models/tattoo.gpickle")