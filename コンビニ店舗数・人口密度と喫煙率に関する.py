#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:57:08 2022

@author: masaokafumiya
"""
## Mac環境で作成
# In[1]:
#ライブラリの読み込み
import numpy as np
import numpy.random as random 
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

#可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()

from sklearn import linear_model

# In[2]:
import os
#ディレクトリの設定
os.chdir('/Users/masaokafumiya/Library/')

# In[3]:
#Excelの読み込み
test_data = pd.ExcelFile('data.xlsx')
print(test_data.sheet_names)
test_data_df = test_data.parse()

# In[4]:
#基本統計量をまとめて吐き出す
##全ての変数を表示
print(test_data_df['コンビニ店舗数（1万人あたり）'].describe())
print(test_data_df['たばこ支出金額'].describe())
print(test_data_df['喫煙率'].describe())
print(test_data_df['人口密度'].describe())

# In[5]:
#地図にプロット(下準備)
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = "Hiragino Maru Gothic Pro"

import cv2
from PIL import Image
import matplotlib.colors
from japanmap import *

test_data_df = test_data_df.iloc[:53,:13]

# In[6]:
#地図にプロット（コンビニ店舗数（1万人あたり））
num_dict1={}

for k,n in zip(test_data_df["都道府県"], test_data_df["コンビニ店舗数（1万人あたり）"]):
    tmp = pref_code(k)
    tmp = pref_names[tmp]
    if tmp not in num_dict1:
        num_dict1[tmp] = n
    else:
        num_dict1[tmp] += n

n_min = min(num_dict1.values())
n_max = max(num_dict1.values())

cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=n_min, vmax=n_max)

def color_scale(r):
    tmp = cmap(norm(r))
    return(tmp[0]*225, tmp[1]*225, tmp[2]*225)

for k,v in num_dict1.items():
    num_dict1[k] = color_scale(v)

plt.figure(figsize=(10,8))
plt.imshow(picture(num_dict1))
plt.grid(False)
plt.title("コンビニ店舗数（1万人あたり）")

sm =plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)
plt.show()

# In[7]:
#地図にプロット(たばこ支出金額)
num_dict2={}

for k,n in zip(test_data_df["都道府県"], test_data_df["たばこ支出金額"]):
    tmp = pref_code(k)
    tmp = pref_names[tmp]
    if tmp not in num_dict2:
        num_dict2[tmp] = n
    else:
        num_dict2[tmp] += n

n_min = min(num_dict2.values())
n_max = max(num_dict2.values())

cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=n_min, vmax=n_max)

def color_scale(r):
    tmp = cmap(norm(r))
    return(tmp[0]*225, tmp[1]*225, tmp[2]*225)

for k,v in num_dict2.items():
    num_dict2[k] = color_scale(v)

plt.figure(figsize=(10,8))
plt.imshow(picture(num_dict2))
plt.grid(False)
plt.title("たばこ支出金額")

sm =plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)
plt.show()

# In[8]:
#地図にプロット（喫煙率)
num_dict3={}

for k,n in zip(test_data_df["都道府県"], test_data_df["喫煙率"]):
    tmp = pref_code(k)
    tmp = pref_names[tmp]
    if tmp not in num_dict3:
        num_dict3[tmp] = n
    else:
        num_dict3[tmp] += n

n_min = min(num_dict3.values())
n_max = max(num_dict3.values())

cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=n_min, vmax=n_max)

def color_scale(r):
    tmp = cmap(norm(r))
    return(tmp[0]*225, tmp[1]*225, tmp[2]*225)

for k,v in num_dict3.items():
    num_dict3[k] = color_scale(v)

plt.figure(figsize=(10,8))
plt.imshow(picture(num_dict3))
plt.grid(False)
plt.title("喫煙率")

sm =plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)
plt.show()

# In[9]:
#地図にプロット(人口密度)
num_dict4={}

for k,n in zip(test_data_df["都道府県"], test_data_df["人口密度"]):
    tmp = pref_code(k)
    tmp = pref_names[tmp]
    if tmp not in num_dict4:
        num_dict4[tmp] = n
    else:
        num_dict4[tmp] += n

n_min = min(num_dict4.values())
n_max = max(num_dict4.values())

cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=n_min, vmax=n_max)

def color_scale(r):
    tmp = cmap(norm(r))
    return(tmp[0]*225, tmp[1]*225, tmp[2]*225)

for k,v in num_dict4.items():
    num_dict4[k] = color_scale(v)

plt.figure(figsize=(10,8))
plt.imshow(picture(num_dict4))
plt.grid(False)
plt.title("人口密度")

sm =plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)
plt.show()

# In[10]:
#複数変数のヒストグラム
sns.pairplot(test_data_df[['コンビニ店舗数（1万人あたり）','たばこ支出金額','喫煙率','人口密度']])
plt.grid(True)

# In[11]:
#相関関係を可視化する（ヒートマップ）
import seaborn as sns
cor = test_data_df[[ 'コンビニ店舗数（1万人あたり）','たばこ支出金額','喫煙率','人口密度']].corr()

sns.heatmap(cor, cmap= sns.color_palette('coolwarm', 10), 
            annot=True,
            fmt='.2f', 
            vmin = -1, 
            vmax = 1
            )
plt.title("相関関係を可視化する（ヒートマップ）")

# In[12]:
#コンビニ店舗数（1万人あたり）と喫煙率の単回帰分析
#グラフの作成
reg =linear_model.LinearRegression()
print(reg)

X = test_data_df.loc[:,['コンビニ店舗数（1万人あたり）']].values
Y = test_data_df['喫煙率'].values

reg.fit(X,Y)

plt.plot(test_data_df['コンビニ店舗数（1万人あたり）'],test_data_df['喫煙率'],"o",)

plt.xlabel('コンビニ店舗数（1万人あたり）')
plt.ylabel('喫煙率')
plt.title("コンビニ店舗数（1万人あたり）と喫煙率の単回帰分析")
plt.grid(True)

plt.plot(X, reg.predict(X))
plt.grid(True)

#回帰係数
print('回帰係数', reg.coef_)
#切片
print('切片', reg.intercept_)
#決定係数
print('決定係数', reg.score(X,Y))
