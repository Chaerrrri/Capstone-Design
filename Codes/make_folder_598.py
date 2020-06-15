import os

CLASS_FILE_598 = 'classes_598.txt'

file = open("classes_598.txt", 'r')
lines = file.readlines()

# make 598 class
folders = []
for line in lines:

    if(len(line) != 7):
        print(line)
    folders.append(line[2:6])

file.close()

from distutils.dir_util import copy_tree

# train folder - 598
for folder in folders:
    copy_tree("./PE92_train/"+folder, "./PE92_train_598/"+folder)

# test folder - 598
for folder in folders:
    copy_tree("./PE92_test/"+folder, "./PE92_test_598/"+folder)
