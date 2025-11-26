import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10, cifar100


(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
#CIFAR 10 

(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data(label_mode='fine')
#CIFAR 100 

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
cifar10_relevant = [1, 2, 3, 4, 5, 7, 9]  

# Required data automobile, bird, cat, deer, dog, horse, truck



cifar100_classes = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
    'lamp','lawn-mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
    'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
    'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

#cifar100_relevant = [19, 34, 2, 11, 35, 76, 38, 57, 84, 27, 8, 13, 48, 58, 92, 61, 99]  

# CIFAR-100 fixed 
cifar100_relevant = [
    19,  # cattle
    34,  # fox
    2,   # baby
    11,  # boy
    35,  # girl
    46,  # man
    98,  # woman
    65,  # rabbit
    80,  # squirrel
    47,  # maple_tree
    52,  # oak_tree
    56,  # palm_tree
    59,  # pine_tree
    96,  # willow_tree
    8,   # bicycle
    13,  # bus
    48,  # motorcycle
    58,  # pickup_truck
    90,  # train
    41,  # lawn-mower
    89   # tractor
]




final_classes = [ # CIFAR-10
                    "automobile", "bird", "cat", "deer", "dog", "horse", "truck",
    # CIFAR-100
    "cattle", "fox", "baby", "boy", "girl", "man", "woman", "rabbit", "squirrel", "trees",
    "bicycle", "bus", "motorcycle", "pickup_truck", "train", "lawn-mower", "tractor"
]

### map each class
# label cifar10 classes
cifar10_to_final = {
    1: 0,  # automobile
    2: 1,  # bird
    3: 2,  # cat
    4: 3,  # deer
    5: 4,  # dog
    7: 5,  # horse
    9: 6   # truck
}

# label cifar100 classes
cifar100_to_final = {
    19: 7,   # cattle
    34: 8,   # fox
    2:  9,   # baby
    11: 10,  # boy
    35: 11,  # girl
    46: 12,  # man
    98: 13,  # woman
    65: 14,  # rabbit
    80: 15,  # squirrel

    # all types of tree
    47: 16,  # maple_tree
    52: 16,  # oak_tree
    56: 16,  # palm_tree
    59: 16,  # pine_tree
    96: 16,  # willow_tree

    8:  17,  # bicycle
    13: 18,  # bus
    48: 19,  # motorcycle
    58: 20,  # pickup_truck
    90: 21,  # train
    41: 22,  # lawn-mower
    89: 23   # tractor
}






def classes_filter(x, y, relevant_index):
    mask = np.isin(y, relevant_index).flatten()
    return x[mask], y[mask]


#Filtering to remove classes that arent used. made with Chatgpt - Luke
# Filter CIFAR-10 to only keep: automobile, bird, cat, deer, dog, horse, truck
x_train10_filt, y_train10_filt = classes_filter(x_train_10, y_train_10, cifar10_relevant)
x_test10_filt,  y_test10_filt  = classes_filter(x_test_10,  y_test_10,  cifar10_relevant)

# Filter CIFAR-100 to only keep the relevant classes (including all tree types)
x_train100_filt, y_train100_filt = classes_filter(x_train_100, y_train_100, cifar100_relevant)
x_test100_filt,  y_test100_filt  = classes_filter(x_test_100,  y_test_100,  cifar100_relevant)

print("CIFAR-10 filtered train shape:", x_train10_filt.shape, y_train10_filt.shape)
print("CIFAR-10 filtered test shape:",  x_test10_filt.shape,  y_test10_filt.shape)
print("CIFAR-100 filtered train shape:", x_train100_filt.shape, y_train100_filt.shape)
print("CIFAR-100 filtered test shape:",  x_test100_filt.shape,  y_test100_filt.shape)

