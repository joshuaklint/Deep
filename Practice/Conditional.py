import os

def add_user():
    confirm = 'N'
    while confirm != 'Y':
        username = input('Enter Username: ')
        print('Is the username correct ' + username + '?Y/N')
        confirm = input().upper()
    os.system('sudo useradd ' + username + ' /add ')

add_user()
