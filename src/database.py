#!/usr/bin/env python3

import pyodbc
server = 'localhost'
database = 'TestDB'
username = 'sa'
password = 'Qwerty123'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

class User(object):
    def __init__(self, id, name):
        self.id = id
        self.name = name
    def get_id(self):
        return self.id
    def get_name(self):
        return self.name

arr_users = []

def test_select_query():
    #Select Query
    print ('Reading data from table')
    tsql = "SELECT id, name FROM userTable;"
    with cursor.execute(tsql):
        row = cursor.fetchone()
        while row:
            print (str(row[0]) + " " + str(row[1]))
            row = cursor.fetchone()

def select_names():
    tsql = "SELECT id, name FROM userTable;"
    with cursor.execute(tsql):
        row = cursor.fetchone()
        while row:
            # print (str(row[0]) + " " + str(row[1]))
            user = User(row[0], row[1])
            arr_users.append(user)
            row = cursor.fetchone()
    
def get_name(id):
    for user in arr_users:
        if str(id) == str(user.get_id()):
            return user.get_name()