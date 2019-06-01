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
ob = Authenticate()
apis = deque(ob.api)

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

def extract_users(user_ids):
    batch_size = 100
    batch_len = math.ceil(len(user_ids) / batch_size)
    df_user_data= pd.DataFrame()
    for idx in tqdm(range(batch_len)):
        batch_users = user_ids[(idx * batch_size):((idx + 1) * batch_size)]
        df_= get_users_batch(batch_users)
        if (not df_.empty):
            df_user_data = pd.concat([df_,df_user_data])
    return df_user_data


def get_users_batch(user_ids):
    apis.rotate(-1)
    api = apis[0]
    df_final = pd.DataFrame()
    try:
        user_ob = api.lookup_users(user_ids)          # batch of user objects returned
        for user_ob_ in user_ob:
            data = get_user_details(user_ob_)
            df_final = pd.concat([data,df_final])
        return df_final
    except tweepy.TweepError as e:
        print(e)
        time.sleep(5)
        df_final = get_user_single(user_ids)
        return df_final

def get_user_single(user_ids):
    apis.rotate(-1)
    api = apis[0]
    df_final = pd.DataFrame()
    for user in user_ids:
        try:
            user_ob = api.get_user([user])
            df_ = get_user_details(user_ob)
            df_final = pd.concat([df_,df_final])
        except tweepy.TweepError as e:
            print(e)
            continue
    return df_final

def get_user_details(user_ob):
    data = pd.DataFrame.from_records(
        [{
            'userID': user_ob.id,
            'screen_name' : user_ob.screen_name,
            'name' : user_ob.name,
            'created_at': user_ob.created_at,
            'location': user_ob.location,
            'followers_count': user_ob.followers_count,
            'friends_count': user_ob.friends_count,
            'listed_count': user_ob.listed_count,
            'favourites_count': user_ob.favourites_count,
            'statuses_count': user_ob.statuses_count,
            'url': user_ob.url,
        }], index=[0])
    return data

if __name__ == '__main__':
    top_dir = get_git_root(os.getcwd())
    input_dir = os.path.join(top_dir,"input")
    df_hexagon_new = pd.read_csv(os.path.join(input_dir,"juul_data.csv"),lineterminator="\n")
    logging.info("extraction of juul data started")
    print("extraction of user handles started")
    user_ids = list(df_hexagon_new.userID.unique())
    user_handles = extract_users(user_ids)
    user_handles.to_csv(os.path.join(input_dir,"user_handles.csv"),index=False)

