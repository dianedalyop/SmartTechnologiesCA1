import numpy as np
import matplotlib.pyplot as plt
 # loading data into exploration file for readability 
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Loaded dataset shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

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


print("PIXEL STATISTICS")  # source Chat-GPT
print("Min:", np.min(X_train))
print("Max:", np.max(X_train))
print("Mean:", np.mean(X_train))
print("Std:", np.std(X_train))
