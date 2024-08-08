import matplotlib.pyplot as plt
import numpy as np
from pyts.classification import LearningShapelets
from pyts.datasets import load_gunpoint
from pyts.utils import windowed_view
import pickle 
import pandas as pd

df = pd.read_excel(r'C:\Users\kamasanis\OneDrive - FM Global\Train.xlsx')
row1 = df['Gearboxload'].values
row2 = df['Gearbox+Load on plate'].values
row3 = df['Faulted rotor bearing'].values
X = np.row_stack((row1, row2, row3))
y = np.array([1, 2, 3])

# Load the data set and fit the classifier
# X, _, y, _ = load_gunpoint(return_X_y=True)
clf = LearningShapelets(random_state=42, tol=0.01,max_iter=10, multi_class='ovo',shapelet_scale=1,n_jobs=-1)
clf.fit(X, y)

pkl_filename = r'C:\Users\kamasanis\OneDrive - FM Global\Downloads\VAG\pickle_model.pkl'  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(clf, file)

# Select two shapelets
shapelets = np.asarray([clf.shapelets_[0, -9], clf.shapelets_[0, -12]])

plt.figure(figsize=(14, 4))

# Plot the two shapelets
plt.subplot(1, 2, 1)
plt.plot(shapelets[0])
plt.plot(shapelets[1])
plt.title('Two learned shapelets', fontsize=14)