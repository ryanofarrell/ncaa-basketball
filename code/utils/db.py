#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:03:02 2019

@author: Ryan
"""
import pymongo
from configparser import ConfigParser


def config(section, filename='database.ini'):
    """Pulls the config status for the given section

    Returns: A dict of parameters for the given section
    """
    # create a parser, read config file
    parser = ConfigParser()
    parser.read(filename)

    # get section
    p = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            p[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return p


def get_db(config_file_name='database.ini'):
    """Opens database connection"""
    #
    # if db is None:
    import os
    prevCwd = os.getcwd()
    os.chdir('/Users/Ryan/Documents/projects/ncaa-basketball/code/utils')
    params = config(section='MONGODB', filename=config_file_name)
    client = pymongo.MongoClient("mongodb+srv://" + params['user'] + ":" + params['password'] + "@basketball-5hnb2.mongodb.net/test?retryWrites=true&w=majority")
    db = client[params['env']]
    print('Connected to database: {}'.format(params['env']))
    os.chdir(prevCwd)

    return db

