import glob
import pandas as pd
import time
import datetime
import xml.etree.ElementTree

path = "data"


def get_timestamp(ts):
    s = ts.split('.')[0]
    return time.mktime(
        datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())


def get_users_df(filename):
    d = {'user_id': 0,
         'reputation': 0,
         'views': 0,
         'upvotes': 0,
         'downvotes': 0,
         'age': 0
         }

    users_df = pd.DataFrame(columns=d.keys())

    i = 0
    root = xml.etree.ElementTree.parse(filename).getroot()
    for row in root:
        if row.attrib['Id'] == "-1":      # deleted users
            continue
        d['user_id'] = row.attrib['Id']
        d['reputation'] = row.attrib['Reputation']
        d['views'] = row.attrib['Views']
        d['upvotes'] = row.attrib['UpVotes']
        d['downvotes'] = row.attrib['DownVotes']
        if 'Age' in row.attrib:
            d['age'] = row.attrib['Age']
        users_df.loc[i] = d
        i = i + 1
    return users_df


def get_posts_df(filename):
    d = {'ques_id': 0,
         'time_to_ans': 0,
         'ans_id': 0,
         'user_id': 0,
         'score': 0,
         'view_count': 0,
         'comment_count': 0,
         'favorite_count': 0
         }

    posts_df = pd.DataFrame(columns=d.keys())

    i = 0
    ans_ts = {}     # temporarily store answers' timestamps
    ques_ts = {}    # temporarily store questions' timestamps
    root = xml.etree.ElementTree.parse(filename).getroot()
    for row in root:
        post_id = row.attrib['Id']
        post_type = row.attrib['PostTypeId']
        if post_type != "1":     # not a question
            if post_type == "2":     # an answer
                ans_ts[post_id] = get_timestamp(row.attrib['CreationDate'])
            continue

        if 'AcceptedAnswerId' not in row.attrib or \
                'OwnerUserId' not in row.attrib or \
                'Id' not in row.attrib:    # doesn't have an AC ans.
            continue

        d['ques_id'] = row.attrib['Id']
        d['ans_id'] = row.attrib['AcceptedAnswerId']
        d['user_id'] = row.attrib['OwnerUserId']
        d['score'] = row.attrib['Score']
        d['view_count'] = row.attrib['ViewCount']
        d['comment_count'] = row.attrib['CommentCount']
        if 'FavoriteCount' in row.attrib:
            d['favorite_count'] = row.attrib['FavoriteCount']

        ques_ts[post_id] = get_timestamp(row.attrib['CreationDate'])

        posts_df.loc[i] = d
        i = i + 1
    # add time_to_ans entries in the dataframe
    for i, row in posts_df.iterrows():
        ques_id = row['ques_id']
        ans_id = row['ans_id']
        q_ts = ques_ts[ques_id]
        a_ts = ans_ts[ans_id]
        posts_df.set_value(i, 'time_to_ans', a_ts - q_ts)
    return posts_df


def handle_users(filename):
    df = get_users_df(filename)
    tokens = filename.split('/')
    tokens[1] = path
    tokens[-1] = "users.csv"
    csv_filename = '/'.join(tokens)
    df.to_csv(csv_filename, sep=',', index=False)
    print "Written ", csv_filename


def handle_posts(filename):
    df = get_posts_df(filename)
    tokens = filename.split('/')
    tokens[1] = path
    tokens[-1] = "posts.csv"
    csv_filename = '/'.join(tokens)
    df.to_csv(csv_filename, sep=',', index=False)
    print "Written ", csv_filename


def main(folder):
    subdir = glob.glob(folder)
    subdir.sort()

    for sub in subdir:
        sub += '/*'
        files = glob.glob(sub)
        filename = 'thefile.xml'
        for filename in files:
            if "users.xml" in filename.lower():
                handle_users(filename)
            if "posts.xml" in filename.lower():
                handle_posts(filename)

if __name__ == '__main__':
    folder = "./extracted/*"
    main(folder)
