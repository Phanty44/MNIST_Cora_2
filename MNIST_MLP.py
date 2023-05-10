import torch
import torchvision
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from torchvision import transforms
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

import visualization

train_data = torchvision.datasets.MNIST(root='~/torch_datasets', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,)),
                                            transforms.Lambda(lambda x: x / 2 + 0.5)
                                        ]))

test_data = torchvision.datasets.MNIST(root='~/torch_datasets', train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize((224, 224)),
                                           transforms.Grayscale(num_output_channels=3),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,)),
                                           transforms.Lambda(lambda x: x / 2 + 0.5)
                                       ]))

vgg = torchvision.models.vgg16(pretrained=True)
print(vgg)
new_classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
vgg.classifier = new_classifier
# Print the new model to verify that the last layer has been removed
print(vgg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

def extract_features(data):
    features = []
    labels = []
    i = 0
    for images, targets in data:
        images = images.unsqueeze(0).to(device)  # Add a batch dimension to the input image and move it to the GPU
        output = vgg(images)  # Pass the padded image through the VGG-like CNN architecture
        features.append(
            output.flatten().detach().cpu().numpy())  # Move the output back to the CPU and convert it to a numpy array
        labels.append(targets)
        i += 1
        if i % 100 == 0:
            print(i)
    return features, labels


print("Extracting features...")
# features, labels = extract_features(train_data)
# x_test, y_test = extract_features(test_data)

# pickle_out = open("X.pickle", "wb")
# pickle.dump(features, pickle_out)
# pickle_out.close()

# pickle_out = open("y.pickle", "wb")
# pickle.dump(labels, pickle_out)
# pickle_out.close()

# pickle_out = open("X_test.pickle", "wb")
# pickle.dump(x_test, pickle_out)
# pickle_out.close()

# pickle_out = open("y_test.pickle", "wb")
# pickle.dump(y_test, pickle_out)
# pickle_out.close()


features = np.array(pickle.load(open("X.pickle", "rb")))
labels = np.array(pickle.load(open("y.pickle", "rb")))
x_test = np.array(pickle.load(open("X_test.pickle", "rb")))
y_test = np.array(pickle.load(open("y_test.pickle", "rb")))

# Create an MLP classifier and fit the model on the training data

# 'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
# 'alpha': [0.0001, 0.001, 0.01],
# 'learning_rate_init': [0.001, 0.01, 0.1],
# 'solver': ["lbfgs","sgd", "adam"],
# 'activation': ['identity', 'logistic', 'tanh', 'relu']
param_grid = {
    'hidden_layer_sizes': [(256, 10)],
    'alpha': [0.0001],
    'learning_rate_init': [0.001],
    'solver': ["adam"],
    'activation': ['relu']
}

clf = MLPClassifier(
    hidden_layer_sizes=(256, 10),
    alpha=0.0001,
    learning_rate_init=0.001,
    solver='adam',
    activation='relu',
    batch_size=100,
    max_iter=10,
    verbose=10,
    random_state=42,
)
# grid_search = GridSearchCV(clf, param_grid, cv=2)
# grid_search.fit(features, labels)
# grid_search.score(x_test, y_test)
# print('Best hyperparameters:', grid_search.best_params_)


print("Predicting...")
clf.fit(features, labels)
val_pred = clf.predict(x_test)
print("Finding accuracy...")
print(classification_report(y_test, val_pred))

visualization.visualize_confusion(y_test, val_pred, clf)
# visualization.visualize_wrong(val_pred, y_test, np.array(test_data.data))
