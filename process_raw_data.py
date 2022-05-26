#!/usr/bin/env python3

"""
The script extracts valuable information from raw Twitter data, mainly meta data is analysed.
There are two main command line arguments to the main function:
- the path to the file with data from particular hour to be processed
- the directory where the result of the processing is to be saved
"""

import gzip
import json
import jsonlines
import numpy as np
import os
import pandas as pd
import sys
import time

from shapely.geometry import mapping, shape
from shapely.prepared import prep
from shapely.geometry import Point

#Extract hashtags, mentions and device sources from underlying data structure, e.g. dictionaries.
def extract_hashtags(tweet_list):
    hashtags=[]
    for hashtag_dict in tweet_list:
        hashtags.append(hashtag_dict['text'])
    return hashtags

def extract_mentions(tweet_list):
    mentions=[]
    for mentions_dict in tweet_list:
        mentions.append(mentions_dict['screen_name'])
    return mentions
    
def extract_urls(tweet_list):
    urls=[]
    for urls_dict in tweet_list:
        urls.append(urls_dict['expanded_url'])
    return urls

def extract_device(tweet):
    if 'iPhone' in tweet or ('iOS' in tweet):
        return 'iPhone'
    elif 'Android' in tweet:
        return 'Android'
    elif 'Mobile' in tweet or ('App' in tweet):
        return 'Mobile device'
    elif 'Mac' in tweet:
        return 'Mac'
    elif 'Windows' in tweet:
        return 'Windows'
    elif 'Bot' in tweet:
        return 'Bot'
    elif 'Web' in tweet:
        return 'Web'
    elif 'Instagram' in tweet:
        return 'Instagram'
    elif 'Facebook' in tweet:
        return 'Facebook'
    elif 'Blackberry' in tweet:
        return 'Blackberry'
    elif 'iPad' in tweet:
        return 'iPad'
    elif 'Foursquare' in tweet:
        return 'Foursquare'
    else:
        return 'Other'


class CountryExtractor(object):
  def __init__(self):
    self.countries = self.load_countries()

  def load_countries(self, filename='countries.geojson'):
    with open(filename) as fid:
      data = json.load(fid)
      
      countries = {}
      for feature in data["features"]:
        geom = feature["geometry"]
        country = feature["properties"]["ISO_A2"] # 'ADMIN' for full name, ISO_A3, ISO_A2 for country codes
        countries[country] = prep(shape(geom))
      #print("#Loaded {} countries from file {}".format(len(countries),filename), file=sys.stderr)
    return countries

  def lonlat_to_country(self, lon, lat):
    point = Point(lon, lat)
    for country, geom in self.countries.items():
      if geom.contains(point):
        return country
    return "??"


def main(path_to_file, save_path):
  tweet_list = []
  count = 0

  #gunzip if necessary
  if path_to_file == "-":
    fid = sys.stdin
  else:
    try:
      fid = gzip.GzipFile(path_to_file)
      fid.peek(0)
    except OSError:
      fid = open(path_to_file)

  country_extractor = CountryExtractor()
  

  # read file line-by-line and parse either as json or as python dictionary
  for line in fid:
          try:
              obj = json.loads(line)
          except json.decoder.JSONDecodeError:
              line = eval(line)
              obj = json.loads(line)
          try:
              id_str = obj['id_str']
              user_id_str = obj['user']['id_str']
              screen_name = obj['user']['screen_name']
              user_created_at = obj['user']['created_at']
              verified = obj['user']['verified']             #true if account of public interest
              location = obj['user']['location']
              followers = obj['user']['followers_count']
              friends = obj['user']['friends_count']
              statuses = obj['user']['statuses_count']       #number of tweets issued from account
              #withheld = obj['user']['withheld_in_countries']
              device = extract_device(obj['source'])

# Check if tweet is a retweet, quote or reply
   
              if 'retweeted_status' in obj:
                is_retweet_of_id_str = obj['retweeted_status']['id_str']
                is_retweet_of_screen_name = obj['retweeted_status']['user']['screen_name']
              else:
                is_retweet_of_id_str = ''
                is_retweet_of_screen_name = ''
                
              if 'quoted_status' in obj:
                is_quote_of_id_str = obj['quoted_status']['id_str']
                is_quote_of_screen_name = obj['quoted_status']['user']['screen_name']
              else:
                is_quote_of_id_str = ''
                is_quote_of_screen_name = ''
                                
              if 'in_reply_to_status_id_str' in obj:
                in_reply_to_id_str = obj['in_reply_to_status_id_str']
                in_reply_to_screen_name = obj['in_reply_to_screen_name']       
              else:
                in_reply_to_id_str = ''
                in_reply_to_screen_name = ''

# Check if any geographical information is provided                   
              
              if obj['coordinates'] is not None:
                coordinates = tuple(obj['coordinates']['coordinates']) 
                country = country_extractor.lonlat_to_country(*coordinates)
              else:
                coordinates = ''
                country = ''

              if obj['place'] is not None:  # not guaranteed to be *from* that location, could also be *about* that location
                  place = obj['place']['full_name']
                  if not country:
                    country = obj['place']['country_code']
                  if not coordinates:
                    bb = obj['place']['bounding_box']['coordinates'][0] # 4-tuple of lon-lat 
                    coordinates = ((bb[0][0]+bb[1][0]+bb[2][0]+bb[3][0])/4.,(bb[0][1]+bb[1][1]+bb[2][1]+bb[3][1])/4.)   # brute-force avg rather than importing numpy
              else:
                  place = ''

              timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(obj['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))


              # at this point, if it's a retweet, we switch to the original content
              
              if is_retweet_of_id_str:
                obj = obj['retweeted_status']

              lang = obj['lang']

              if 'extended_tweet' in obj:
                  text = obj['extended_tweet']['full_text']
                  hashtags = obj['extended_tweet']['entities']['hashtags']
                  mentions = obj['extended_tweet']['entities']['user_mentions']
                  urls = obj['extended_tweet']['entities']['urls']
              else: 
                  text = obj['text']
                  hashtags = obj['entities']['hashtags']
                  mentions = obj['entities']['user_mentions']
                  urls = obj['entities']['urls']
              text = text.replace('\n',' ').replace('\r',' ') # remove annoying line breaks in text
              
              # Transform format of hashtags, mentions and urls for better readability
              hashtags = extract_hashtags(hashtags)
              mentions = extract_mentions(mentions)
              urls = extract_urls(urls)


# Append all extracted features to list for storing
              tweet_list.append((id_str,
                                 timestamp,
                                 text, len(text),
                                 lang,
                                 user_id_str,
                                 screen_name,
                                 user_created_at,
                                 verified,
                                 followers,
                                 friends,
                                 statuses,
                                 device,
                                 location,
                                 coordinates, 
                                 country,
                                 place, 
                                 is_retweet_of_id_str,
                                 is_retweet_of_screen_name,
                                 is_quote_of_id_str,
                                 is_quote_of_screen_name,
                                 in_reply_to_id_str,
                                 in_reply_to_screen_name,
                                 obj['retweet_count'],
                                 obj['quote_count'],
                                 obj['reply_count'],
                                 obj['favorite_count'],
                                 hashtags,
                                 mentions,
                                 urls
                                ))
          except(KeyError): # e.g. no bounding box, even though there should be or connection error
              count+=1
              pass

  fid.close() 

  #Convert list of tweets to dataframe
   df_tweets = pd.DataFrame(tweet_list, 
                           columns=['id', 'created_at', 'text', 'length', 'lang', 'user_id', 
                                    'user_screen_name', 'user_created_at', 'verified', 'followers', 
                                    'friends', 'statuses', 'source', 'location', 'coordinates', 'country', 
                                    'place', 'is_retweet_of_id', 'is_retweet_of_screen_name', 'is_quote_of_id', 
                                    'is_quote_of_screen_name', 'in_reply_to_id', 'in_reply_to_screen_name', 'retweet_count', 
                                    'quote_count', 'reply_count', 'favorite_count', 'hashtags', 'mentions', 'urls'])
                                    
  # Remove .gz and .jsonl extensions and save results as json
  if save_path == "-":
      df_tweets.to_json(sys.stdout, orient='records', lines=True)
  else:
      save_path_file = save_path+os.path.basename(path_to_file)
      if not os.path.exists(save_path):
          os.makedirs(save_path)
      save_path_file = save_path_file.replace('.gz', '').replace('.jsonl', '').replace('.json','').replace('.csv', '') + ".ex.jsonl"
      df_tweets.to_json(save_path_file, orient='records', lines=True)


if __name__ == "__main__":
   try:
    path_to_file = sys.argv[1]
  except IndexError:
    path_to_file = '-'
   try:
    save_path = sys.argv[2]
  except IndexError:
    save_path = '-'

  main(path_to_file, save_path)
