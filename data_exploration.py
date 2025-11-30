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

print(" CHANNEL STATS")
R_mean = np.mean(X_train[:,:,:,0])
G_mean = np.mean(X_train[:,:,:,1])
B_mean = np.mean(X_train[:,:,:,2])
R_std  = np.std(X_train[:,:,:,0])
G_std  = np.std(X_train[:,:,:,1])
B_std  = np.std(X_train[:,:,:,2])

print(f"Red mean: {R_mean:.4f}, std: {R_std:.4f}")
print(f"Green mean: {G_mean:.4f}, std: {G_std:.4f}")
print(f"Blue mean: {B_mean:.4f}, std: {B_std:.4f}")

plt.figure(figsize=(6,4))
plt.bar(["R","G","B"], [R_mean, G_mean, B_mean])
plt.title("Average Pixel Intensity per Channel")
plt.tight_layout()
plt.show()

print("BRIGHTNESS & CONTRAST DISTRIBUTION") # Diane wt Copilot
brightness = np.mean(X_train, axis=(1,2,3))
contrast = np.std(X_train, axis=(1,2,3))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.hist(brightness, bins=40)
plt.title("Brightness Distribution (mean pixel)")
plt.xlabel("Brightness")
plt.ylabel("Count")

plt.subplot(1,2,2)
plt.hist(contrast, bins=40)
plt.title("Contrast Distribution (std dev)")
plt.xlabel("Contrast")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

print("AVERAGE IMAGE / CLASS")
num_classes = len(unique)
cols = 5
rows = int(np.ceil(num_classes / cols))
plt.figure(figsize=(cols*2.2, rows*2.2))
for idx, cls in enumerate(unique):
    imgs = X_train[y_train == cls]
    avg_img = np.mean(imgs, axis=0)
    plt.subplot(rows, cols, idx+1)
    plt.imshow(avg_img)
    plt.title(f"Class {cls}")
    plt.axis("off")
plt.suptitle("Average Image per Class")
plt.tight_layout()
plt.show()

print("STD-DEV IMAGE PER CLASS")
plt.figure(figsize=(cols*2.2, rows*2.2))
for idx, cls in enumerate(unique):
    imgs = X_train[y_train == cls]
    std_img = np.std(imgs, axis=0)
    plt.subplot(rows, cols, idx+1)
    plt.imshow(std_img)
    plt.title(f"Std Class {cls}")
    plt.axis("off")
plt.suptitle("Std-dev Image per Class ")
plt.tight_layout()
plt.show()

