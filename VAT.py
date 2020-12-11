#!/usr/local/Cellar python
# -*- coding: utf-8 -*-
"""
@File Name: ensemble.py
@Author: 姜小帅
@Motto: 良好的阶段性收获是坚持的重要动力之一
@Date: 2020/2/19
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering


class VAT:
    """
    ########
    Input: 
    @similarity_matrix：matrix of similarity of sample
    ########
    Output: the image of reordered similarity matrix of sample. (The more similar the closer)

    @Example:
    mat = np.array([
        [0, 0.73, 0.19, 0.71, 0.16], 
        [0.73, 0, 0.59, 0.12, 0.78], 
        [0.19, 0.59, 0, 0.55, 0.19], 
        [0.71, 0.12, 0.55, 0, 0.74], 
        [0.16, 0.78, 0.19, 0.74, 0]
        ])
    vat = VAT(mat)
    vat.plot()
    """

    def __init__(self, similarity_matrix):
        self.dissimilarity_matrix = 1 - similarity_matrix

    def plot(self):
        P = []
        I = set()
        K = set(range(len(self.dissimilarity_matrix)))
        J = K

        max_element_row = np.unravel_index(self.dissimilarity_matrix.argmax(), self.dissimilarity_matrix.shape)[0]
        P.append(max_element_row)
        I.add(max_element_row)
        J = J.difference(I)

        while len(J) != 0:
            m = np.array(list(I)).reshape(len(I), 1)
            n = np.array(list(J)).reshape(len(J), 1)
            temp = self.dissimilarity_matrix[m, n.T]

            min_element_col = np.unravel_index(temp.argmin(), temp.shape)[1]
            I.add(n[min_element_col][0])
            P.append(n[min_element_col][0])
            J = J.difference(I)

        ordered_array = self.dissimilarity_matrix[:, P][P, :]

        plt.imshow(ordered_array, cmap="bone")
        plt.show()
    