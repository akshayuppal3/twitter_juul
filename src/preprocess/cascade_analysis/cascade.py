from authentication import Authenticate
import pandas as pd
import logging
import tweepy
from tqdm import tqdm
import networkx as nx
import git
import util
from collections import deque
import os
import numpy as np
logging.basicConfig(level=logging.INFO, format= util.format, filename= os.path.join(util.logdir,"cascade.log"))


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
		logging.info(str("finding connection for " + str(typef) + " network might take some time"))
		if (util.is_number(user_list)):
			user_list = list([user_list])
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
				df = df.append(data, ignore_index=True)
			except tweepy.TweepError as e:
				continue
		return df

	# @param: user_list= list of users ,df: (dataframe) containing user information
	# @ return a G with node attributes (# friends, # followers, # level)
	def get_node_attributes(self,G,user_list,df,level,source_node=None):
		attr = dict()
		if (util.is_number(user_list)):
			user_list = list([user_list])
		if user_list:   # can't label blank user list
			if (source_node != None):
				if (source_node in user_list):
					user_list.remove(source_node)
				a = df.loc[df.userID == source_node].head(1)
				attr_source = {source_node : {'level': 0,
											 'friends' : list(a['friendsCount'])[0],
											 'followers' : list(a['friendsCount'])[0]}}
				nx.set_node_attributes(G,attr_source)
			for user in user_list:
				if user in list(df.userID):
					user_data = df.loc[df.userID == user].head(1)
					if (user in G.nodes):
						if ('level' not in G[user]):    # if the level doesn't exist already
							attr[(user)] = {'friends': list(user_data['friendsCount'])[0],
							                'followers': list(user_data['userFollowersCount'])[0],
							                'level': level}
			nx.set_node_attributes(G,attr)
		return G

	# @param source node, user_list and dataframe
	# now will check the relationship of all node with the source node, does it have folower or following rel.
	def create_cascade_lvl_1(self,source_node, user_list,):
		G = nx.DiGraph()  # will add edges directly
		# users type(int)
		first_nodes = list()
		if (util.is_number(user_list)):
			user_list = list([user_list])
		for user in tqdm(user_list):
			apis = deque(self.api_list)
			apis.rotate(-1)
			api = apis[0]
			if (user != source_node):
				try:
					relation_obj = api.show_friendship(source_id=(source_node), target_id=(user))[0]
					if ((relation_obj.following == True) or (relation_obj.followed_by == True)):
						if (relation_obj.followed_by == True):
							G.add_edge(user, source_node)
							if user not in first_nodes:
								first_nodes.append(user)
				except tweepy.TweepError as e:
					continue
		return (G, first_nodes)

	# get the next level of cascades...
	def create_cascade(self,G, source_id, user_list):
		if (util.is_number(user_list)):
			user_list = list([user_list])
		if (util.is_number(source_id)):
			source_id = list([source_id])
		if (len(user_list) != 0):
			second_user = list()
			# find the follower relationship
			df_followers = self.find_connections( source_id, 'followers')
			if (not df_followers.empty):
				for node in tqdm(source_id):  # node : source_id list
					for user in user_list:    # user : user_list
						if ('followers_list' in df_followers):
							if (node in list(df_followers.userID)):
								followers = (df_followers.followers_list[df_followers.userID == node].values[0])
								if (user in set(followers)):
									second_user.append(user) # if node not in G.nodes
									G.add_edge(user, node)
				second_user = list(set(second_user))
				rem_users = list(set(user_list) - set(second_user))
				return (G, second_user, rem_users)
			else:
				return (G, source_node, user_list)
		else:
			return (G,source_id,user_list)

	def get_cascade(self,df, source_node, user_list, level_termiante=None):
		if (util.is_number(user_list)):
			user_list = list([user_list])
		G, first_users = self.create_cascade_lvl_1(source_node, user_list)
		# rest levels
		if source_node in first_users:
			first_users.remove(source_node)
		G = self.get_node_attributes(G, first_users, df, 1, source_node=source_node)
		rem_users = set(user_list) - set(first_users)
		level = 2  ## for the next levels
		users_next = first_users
		if(len(G.nodes()) > 0):
			if (rem_users):  # there rem users to continue to next level and G should not be empty
				while True:
					G, users_next, rem_users = self.create_cascade(G, users_next, rem_users)
					G = self.get_node_attributes(G, users_next, df, level)
					print("at level",level)
					logging.info(str("at level "+ str(level)))
					level += 1
					if (not rem_users) or (not users_next): # no more remaining users
						return G
					if level_termiante:
						if (level > level_termiante):
							return G
		else:
			return (G)

	## get existing files
	def get_existing_user(self,path):
		df_users = util.readCSV(path)
		users = util.getUsers(df_users,'ID')
		return users

	## getting the unique tweets with retweet count> 0
	## @params input data containing tweets
	## @return tweets
	def get_unique_tweets(self,df)->pd.DataFrame():
		tweet_text_list = list()
		df_tweets = pd.DataFrame([])
		for index, tweet in df.iterrows():
			text = tweet['tweetText']
			retweet_count = tweet['retweetCount']
			if retweet_count > 0:
				if text not in tweet_text_list:
					tweet_text_list.append(text)
					df_tweets = df_tweets.append(pd.DataFrame({'tweet_text': text,
					                                           'retweet_count': retweet_count}, index=[0]),
					                             ignore_index=True)
		return df_tweets

if __name__ == '__main__':
	logging.info("*****************new extraction of cascade process started***********************")
	hexagon_path = os.path.join(util.get_git_root(os.getcwd()),"input","juul_data.csv")
	model_path = os.path.join(util.get_git_root(os.getcwd()), "models")
	# explored_cascade_path = os.path.join(util.get_git_root(os.getcwd()), "models",'cascade_explored.csv')
	# get the existing files:
	# existing_users = cas.get_existing_user(explored_cascade_path) # not using
	hexagon_data = pd.read_csv(hexagon_path, lineterminator="\n")
	cas = Cascade()
	df_tweets = cas.get_unique_tweets(hexagon_data)
	for i in range(len(df_tweets)):
		cascade = hexagon_data.loc[hexagon_data.tweetText == df_tweets.tweet_text[i]]
		cascade['tweetCreatedAt'] = pd.to_datetime(cascade['tweetCreatedAt'])
		cascade.sort_values(by='tweetCreatedAt', ascending=True, inplace=True)
		source_node = cascade.head(1)['userID'].values[0]
		if (source_node):                                           #if (source_node not in set(existing_users)):
			logging.info(str("creating cascade for user " + str(source_node)))
			retweet_count = cascade.head(1)['retweetCount'].values[0]
			users = set(list(cascade['userID'])) # to remove duplicate entries
			users = list(users)
			if (source_node in users):
				users.remove(source_node)
			if (isinstance(users, (int, np.integer))):
				user_list = list([users])
				if(set([users]) == set([source_node])):
					continue    # if both the source and users are same then don't create cascade
			if (source_node and users):
				G = cas.get_cascade(cascade, source_node, users, level_termiante=None)
				if (G):   # don't dump blank graphs
					if (G.nodes != 0):
						filename = str('G_' + str(source_node) + '_' + str(retweet_count) + '.gpickle')
						nx.write_gpickle(G, os.path.join(model_path, 'graphs', filename))
				else:
					logging.info(str("userID: " + str(source_node) + " no cascade returned"))
		else:
			logging.info(str("userID: " + str(source_node) + " already exists"))

