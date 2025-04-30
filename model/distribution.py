import matplotlib.pyplot as plt
import numpy as np

labels = np.load("../data/labels.npy")

label_freq = {}
for label in labels:
    if label in label_freq:
        label_freq[label] += 1
    else:
        label_freq[label] = 1

sorted_labels = sorted(label_freq.items(), key=lambda pair: pair[1], reverse=True)

label_names, frequencies = zip(*sorted_labels)

print("Number of classes:", len(label_names))
print("Number of data points:", len(labels))

plt.bar(label_names, frequencies)
plt.show()
