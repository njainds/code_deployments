import tornado.ioloop
import tornado.web
import tornado.escape
import tornado.options
import json
import Settings
from pathlib import Path
# import requests
import warnings
import sqlite3
# warnings.filterwarnings('ignore')
import pandas as pd
import string
import random
import pandas as pd
import numpy as np
import gc
import os
import re
import keras
import pickle

from keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, GlobalMaxPool1D, GlobalMaxPooling1D, \
    concatenate, SpatialDropout1D
from keras.models import Model, load_model, model_from_json
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import callbacks
# ==========================
# KB

def  update_icd(input: str, output: str, user_confirm: str):
    connection = create_connection('icd.db')
    if connection == None:
        print('Can not establish the connection to icd.db')
        return -1
#create table
    sql_create_icd_table = """ CREATE TABLE IF NOT EXISTS icd (
                                        input text NOT NULL,
                                        output text NOT NULL,
                                        user_confirm text NOT NULL
                                    ); """
    try:
        c = connection.cursor()
        c.execute(sql_create_icd_table)                   
    except sqlite3.Error as e:
        print (e)    
        return -1
#insert into icd
    sql_insert_icd_table = "INSERT INTO icd (input,output,user_confirm) VALUES"
    sql_insert_icd_table_data =   [(input, output, user_confirm)]
    print(sql_insert_icd_table_data)

    
    try:    
        curs = connection.cursor()
        curs.executemany('insert into icd values (?,?,?)', sql_insert_icd_table_data)  
        connection.commit()
        connection.close()
        # curs.commit()
    except sqlite3.Error as e:
        print (e)
        return -1
    return None    

###############################################################################
def create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return None

loaded_models = []
def load_KB():
    mispell_dict = np.load(os.getcwd() + '/datafile/mispell_dict.npy', allow_pickle=True).item()
    icd_dict = np.load(Path(os.getcwd()) / 'datafile' / 'icd_dict.npy', allow_pickle=True).item()
    word_index = np.load(Path(os.getcwd()) / 'datafile' / 'word_index.npy', allow_pickle=True).item()
    return mispell_dict, icd_dict, word_index
#globlal variables
mispell_dict, icd_dict, word_index = load_KB()
itoicd = dict((v, k) for k, v in icd_dict.items())
with open(Path(os.getcwd()) / 'datafile' / 'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#list of json model
for i in range(5):
    model_name = 'model' + str(i) + '.json'
    json_file = open(Path(os.getcwd()) / 'models' / model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    modelh_name = 'model' + str(i) + '.h5'
    loaded_model.load_weights(Path(os.getcwd()) / 'models' / modelh_name)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
#add model to list
    loaded_models.append(loaded_model)        
#----------------------------------------------------------------
def clean(text):
    x = text.lower()
    x = x.replace(r'+ve', ' positive ')
    x = x.replace(r'+', ' positive ')
    x = re.sub(r"\b([l][.])", "left ", x)
    x = re.sub(r"\b([r][.])", "right ", x)
    x = re.sub(r"(?<=\d)(st|nd|rd|th)\b", '', x)
    x = ((''.join('#' + i + '#' if i.isdigit() else i for i in x)).replace('#/#', '<>')).replace('#', '')
    x = x.replace(r" rt l", " right lower")
    x = x.replace(r" right l", " right lower")
    x = x.replace(r" le l", " left lower")
    x = x.replace(r" lft l", " left lower")
    x = x.replace(r"\n", r" ")
    x = x.replace(r"\t", r" ")
    x = x.replace(r"\b", r" ")
    x = x.replace(r"\r", r" ")
    x = re.sub(r"\s+", r" ", x)
    x = re.sub(r'[\?\.\!\,\=]+(?=[\?\.\!\,\=])', '', x)
    toks = re.split(' |;|,|\*|\n|[(]|[)]|/|[+]|:|-', x)
    ltok = [mispell_dict[tok] if mispell_dict.get(tok) is not None else tok for tok in toks]
    x = [word_index[k] if word_index.get(k) is not None else 1 for k in ltok]
    x = x[:7]
    x = np.array([[0] * (7 - len(x)) + x if len(x) < 7 else x])
    return x
def text_predict(str_text):
    x_test = clean(str_text)
    partial_scores = []
    score = 0
    for i in range(5):
        print(len(loaded_models))
        loaded_model = loaded_models[i]
        score = score + loaded_model.predict(x_test, batch_size=1, verbose=1) / 5
        
    probs = np.sort(score[0])
    probs = probs[-5:]
     
    print(probs)     
    score = np.argsort(score[0])    
    print(score[-5:])
    score = score[-5:]
     
    for i in range(5):
        i = 4 - i
        score_i = probs[i] *100 
        partial_scores.append(list([float(itoicd[score[i]]), float("{0:.2f}".format(score_i))]))
        
    return partial_scores
class Predicttext(tornado.web.RequestHandler):
    def post(self):
        input_text = tornado.escape.json_decode(self.request.body)
        print('---> these are some things that client have sent: ',input_text)
        # response= "aaaaaa" cardiovascular
        results =text_predict(input_text)
        json_file = {
            "data" : results
        }
        self.write(json_file)

class Updata_database(tornado.web.RequestHandler):
    def post(self):
        input_text = tornado.escape.json_decode(self.request.body)
        
        print(' update-data',input_text['data'][0])
        update_icd(str(input_text['data'][0]),str(input_text['data'][1]),str(input_text['data'][2]))
        # response= "aaaaaa" cardiovascular
        
        self.write("update database done")

# ============= login ============
class BaseHandler(tornado.web.RequestHandler):
    def get_current_user(self):
        return self.get_secure_cookie("user")

class MainHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        username = tornado.escape.xhtml_escape(self.current_user)
        # self.write("web_application.html")
        self.render("index.html")


class AuthLoginHandler(BaseHandler):
    def get(self):
        try:
            errormessage = self.get_argument("error")

        except:
            errormessage = ""
        self.render("login.html", errormessage=errormessage)

    def check_permission(self, password, username):
        if username == "admin" and password == "fsoft@12345":
            return True
        return False

    def post(self):
        username = self.get_argument("username", "")
        password = self.get_argument("password", "")
        auth = self.check_permission(password, username)
        if auth:
            self.set_current_user(username)
            # self.redirect(self.get_argument("next", u"/"))
            self.render("index.html")
        else:
            error_msg = u"?error=" + tornado.escape.url_escape("Login incorrect")
            self.redirect(u"/auth/login/" + error_msg)

    def set_current_user(self, user):
        if user:
            self.set_secure_cookie("user", tornado.escape.json_encode(user), expires_days=0.003)
            # print('aaa')
        else:
            self.clear_cookie("user")


class AuthLogoutHandler(BaseHandler):
    def get(self):
        self.clear_cookie("user")
        self.redirect(self.get_argument("next", "/"))

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
                    (r"/", MainHandler),
                    (r'/auth/login/', AuthLoginHandler),
                    (r'/auth/logout/', AuthLogoutHandler),
                    (r"/updata_database", Updata_database),
                    (r"/predict_text",Predicttext),
                    (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': './static'})
                    ]
        settings = {
                        "template_path": Settings.TEMPLATE_PATH,
                        "static_path": Settings.STATIC_PATH,
                        "debug": Settings.DEBUG,
                        "cookie_secret": Settings.COOKIE_SECRET,
                        "xsrf_cookies": False,
                        "login_url": "/auth/login/"
                    }
        tornado.web.Application.__init__(self, handlers, **settings)

def make_app():
    return tornado.httpserver.HTTPServer(Application())


if __name__ == '__main__':
    app = make_app()
    # app = make_app()
    app.listen(5000)
    tornado.ioloop.IOLoop.current().start()


##############################################################################








