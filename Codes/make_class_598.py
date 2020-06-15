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

file2 = open("files_598.txt", 'w')

for folder in folders:
    file2.write(folder)
    file2.write('\n')

file2.close()
