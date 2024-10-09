import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

data = {"radius_mean":14.127291739894552,
        "texture_mean":19.289648506151142,
        "perimeter_mean":91.96903339191564,
        "area_mean":654.8891036906855,
        "smoothness_mean":0.0963602811950791,
        "compactness_mean":0.10434098418277679,
        "concavity_mean":0.0887993158172232,
        "concave points_mean":0.04891914586994728,
        "symmetry_mean":0.18116186291739894,
        "fractal_dimension_mean":0.06279760984182776,
        "radius_se":0.40517205623901575,
        "texture_se":1.2168534270650264,
        "perimeter_se":2.8660592267135327,
        "area_se":40.337079086116,
        "smoothness_se":0.007040978910369069,
        "compactness_se":0.025478138840070295,
        "concavity_se":0.03189371634446397,
        "concave points_se":0.011796137082601054,
        "symmetry_se":0.02054229876977153,
        "fractal_dimension_se":0.0037949038664323374,
        "radius_worst":16.269189806678387,
        "texture_worst":25.677223198594024,
        "perimeter_worst":107.26121265377857,
        "area_worst":880.5831282952548,
        "smoothness_worst":0.13236859402460457,
        "compactness_worst":0.25426504393673116,
        "concavity_worst":0.27218848330404216,
        "concave points_worst":0.11460622319859401,
        "symmetry_worst":0.2900755711775044,
        "fractal_dimension_worst":0.0839458172231986}
cat = []
for i in data:
    cat.append(i)

print(cat)