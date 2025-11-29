import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam



from tensorflow.keras.datasets import cifar10, cifar100



(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
#CIFAR 10 

(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data(label_mode='fine')
#CIFAR 100 

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
cifar10_relevant = [1, 2, 3, 4, 5, 7, 9]  #automobile, bird, cat, deer, dog, horse, truck


# Required data automobile, bird, cat, deer, dog, horse, truck
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




def remap_labels(y, mapping_dict):
    """
    y: label array (N x 1)
    mapping_dict: dict from original label -> new label (0..23)
    """
    y = y.flatten()
    new_y = np.array([mapping_dict[int(label)] for label in y])
    return new_y



# Remap CIFAR-10 labels
y_train10_final = remap_labels(y_train10_filt, cifar10_to_final)
y_test10_final  = remap_labels(y_test10_filt,  cifar10_to_final)

# Remap CIFAR-100 labels
y_train100_final = remap_labels(y_train100_filt, cifar100_to_final)
y_test100_final  = remap_labels(y_test100_filt,  cifar100_to_final)



#Below code to combine the filtered and mapped data. Created in chatgpt - Luke
# Combine training sets
X_train = np.concatenate([x_train10_filt, x_train100_filt], axis=0)
y_train = np.concatenate([y_train10_final, y_train100_final], axis=0)
# Combine test sets
X_test = np.concatenate([x_test10_filt, x_test100_filt], axis=0)
y_test = np.concatenate([y_test10_final, y_test100_final], axis=0)

print("Combined training set:", X_train.shape, y_train.shape)
print("Combined test set:", X_test.shape, y_test.shape)


# Normalizing pixel values range (0, 1) - Diane
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

print("Normalized training :", X_train.shape)
print("Normalized test :", X_test.shape)


shuffle_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

#conversion to one-hot encode - ChatGpt -Diane
num_classes = 24
y_train_oneh = keras.utils.to_categorical(y_train, num_classes)
y_test_oneh  = keras.utils.to_categorical(y_test, num_classes)

print("One-hot labels:", y_train_oneh.shape, y_test_oneh.shape)

#Data Exploration PART 2

#shapes
print("Shapes")
print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)
print(" ")


print("IMAGE GRID confirms labels match images ") 
plt.figure(figsize=(12,6))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(f"Class {y_train[i]}")
    plt.axis("off")
plt.show()

print("CLASS DISTRIBUTION")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c} samples")

 

unique, counts = np.unique(y_train, return_counts=True)

plt.figure(figsize=(12, 6))
plt.bar(unique, counts)

plt.title("Distribution of class in Training Dataset", fontsize=14)
plt.xlabel("Class Index (0 → 23)", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)

plt.xticks(unique)  
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()

print("TRAIN vs TEST CLASS DISTRIBUTION ")
unique_test, counts_test = np.unique(y_test, return_counts=True)

plt.figure(figsize=(10,5))
plt.plot(unique, counts, label="Train")
plt.plot(unique_test, counts_test, label="Test")
plt.title("Training vs Test Class Counts")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.legend()
plt.show()




##Building the model - Luke
num_classes=24

def build_cnn_model(num_classes):
    model = Sequential()   
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 3), activation='relu'))   #convolutional layers
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))   #make images smaller for faster training
    model.add(Conv2D(30, (3, 3), activation='relu'))  #convolutional layers
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #make images smaller again
    model.add(Flatten()) #flatten map
    model.add(Dense(500, activation='relu'))      # a connected layer with 500 neurons.
    model.add(Dropout(0.5))     # turns 50% of neurons off during training. to reduce overfitting.
    model.add(Dense(num_classes, activation='softmax')) #final output layer.
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])     #prepare model for training.

    return model



#Training the model (Chatgpt - Luke)
val_fraction = 0.2     # use 20 percent of training data for validation
num_train = int((1 - val_fraction) * X_train.shape[0])
X_train_part = X_train[:num_train]
y_train_part = y_train_oneh[:num_train]
X_valid = X_train[num_train:]
y_valid = y_train_oneh[num_train:]
print("Train subset shape:", X_train_part.shape, y_train_part.shape)
print("Validation subset shape:", X_valid.shape, y_valid.shape)



def evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    print(model.summary())
    #Train model
    history = model.fit(X_train, y_train, epochs=10, batch_size=400, validation_data=(X_valid, y_valid), verbose=1, shuffle=1)
    #Plot training vs validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.show()
    # Plot training vs validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()
    # Evaluate on the test set
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


##Build the model and training and evaluation. Luke.
model = build_cnn_model(num_classes)
evaluate_model(model, X_train_part, y_train_part, X_valid, y_valid, X_test, y_test_oneh)