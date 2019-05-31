import os 
import pandas as pd
from authentication import Authenticate
from collections import deque
import math
import tweepy
import git
from tqdm import tqdm
import time
import util
import logging

logging.basicConfig(level="INFO", format= util.format, filename=(util.logdir + "/user_scrpaing.log"))
    
def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

def get_users_details(user_ids):
    batch_size = 100
    batch_len = math.ceil(len(user_ids) / batch_size)
    df_user_data= pd.DataFrame()
    print("extracting user handles")
    for idx in tqdm(range(batch_len)):
        apis.rotate(-1)
        api = apis[0]
        batch_users = user_ids[(idx * batch_size):((idx + 1) * batch_size)]
        try:
            user_ob = api.lookup_users(batch_users)
            df_= get_users_details(user_ob)
            df_user_data = pd.concat([df_,df_user_data])
        except tweepy.TweepError as e:
            print(e)
            time.sleep(5)
            user_ob = api.lookup_users(batch_users)
            df_ = get_users_details(user_ob)
            df_user_data = pd.concat([df_, df_user_data])
            continue
    return df_user_data

def get_user_obj(obj):
    df_user_data = []
    for user_ob in obj:
        user_id = user_ob.id,
        screen_name = user_ob.screen_name
        location = user_ob.location
        followers_count = user_ob.followers_count
        followersCount = user_ob.user.followers_count,
        listed_count = user_ob.user.listed_count,
        favourites_count = user_ob.favourites_count,
        statuses_count = user_ob.statuses_count,
        created_at = user_ob.created_at,
        df_user_data.append([user_id,screen_name,location,followers_count,listed_count,followersCount,favourites_count,statuses_count,created_at])
    return pd.DataFrame(df_user_data,columns=["userID","screen_name","location",
                                              "followers_count","listed_count",
                                              "favourites_count","statuses_count","created_at"])

if __name__ == '__main__':
    ob = Authenticate()
    apis = deque(ob.api)
    top_dir = get_git_root(os.getcwd())
    input_dir = os.path.join(top_dir,"input")
    df_hexagon_new = pd.read_csv(os.path.join(input_dir,"juul_data.csv"),lineterminator="\n")
    user_ids = list(df_hexagon_new.userID.unique())
    user_handles = get_users_details(user_ids)
    user_handles.to_csv(os.path.join(input_dir,"user_handles.csv"),index=False)

