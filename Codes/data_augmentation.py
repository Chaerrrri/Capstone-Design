#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[23]:


train_folder_path = './Desktop/chaeri/2020_1/capstone/data/PE92_train_598_aug/'
categories = []

with open('./Desktop/chaeri/2020_1/capstone/data/PE92_data/files_598.txt', 'r') as f:
    infoFile = f.readlines()
    
    
    for line in infoFile:
        words = line.split()
        categories.append(words[0])
    
num_classes = len(categories)


# In[24]:


print(num_classes)


# In[26]:


datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        )

for idex, category in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = train_folder_path + category + '/'
    
    for top, dir, f in os.walk(image_dir):
        
        image_paths = []
        
        for filename in f:
            image_path = image_dir+filename
            image_paths.append(image_path)
        
        for image_p in image_paths[:30]:
            img = load_img(image_p, grayscale=True, color_mode="grayscale")
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=image_dir, save_prefix='힣_Augmented', save_format='png'):
                i += 1
                if i >= 1:
                    break  

