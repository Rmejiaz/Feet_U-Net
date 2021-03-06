import numpy as np
import matplotlib.pyplot as plt
from utils import DiceSimilarity, DiceImages
from sklearn.metrics import jaccard_score

X = plt.imread("Dataset_Unificado/Test/BinaryMasks/maryuri_alvarez_66660829_t5.png")

Dice = DiceSimilarity(X,X)

print(f"Dice similarity of the same mask: {Dice}")

Y = plt.imread("Dataset_Unificado/Test/BinaryMasks/maryuri_alvarez_66660829_t10.png")

Dice2 = DiceSimilarity(X,Y)

print(f"Dice similarity of two similar masks {Dice2}")


Dice3 = DiceImages("./Dataset_Unificado/Test/BinaryMasks/maryuri_alvarez_66660829_t10.png","./Dataset_Unificado/Test/BinaryMasks/maryuri_alvarez_66660829_t5.png")

print(f"Using DiceImages: {Dice3}")


Jaccard = jaccard_score(X,X)

print(f"Jaccard score of the same mask: {Jaccard}")

Jaccard2 = jaccard_score(X,Y)

print(f"Jaccard score of two different masks: {Jaccard2}")
