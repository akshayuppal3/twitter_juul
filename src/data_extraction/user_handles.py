import os 
import pandas as pd
from authentication import Authenticate
from collections import deque
import math
import tweepy
import git
from tqdm import tqdm
    
def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

if __name__ == '__main__':
    ob = Authenticate()
    apis = deque(ob.api)
    top_dir = get_git_root(os.getcwd())
    input_dir = os.path.join(top_dir,"input")
    df_hexagon_new = pd.read_csv(os.path.join(input_dir,"juul_data.csv"),lineterminator="\n")
    user_ids = list(df_hexagon_new.userID.unique())
    batch_size = 100
    batch_len = math.ceil(len(user_ids)/batch_size)
    user_handles = []
    print("extracting user handles")
    for idx in tqdm(range(batch_len)):
        apis.rotate(-1)
        api = apis[0]
        batch_users = user_ids[(idx*batch_size):((idx+1)*batch_size)]
        data = api.lookup_users(batch_users)
        for user_ob in data:
            user_handles.append(user_ob.screen_name)
    user_handles = pd.DataFrame(user_handles,columns=["user_handle"])
    user_handles.to_csv(os.path.join(input_dir,"user_handles.csv"),index=False)

