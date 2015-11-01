# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:45:00 2015

@author: Pankaj
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy as sp
from scipy import linalg
from scipy import sparse as sps
from scipy.sparse import linalg as spla
from apgl.graph import *
from apgl.generator.KroneckerGenerator import KroneckerGenerator

initialGraph = SparseGraph(VertexList(5, 1))
initialGraph.addEdge(1, 2)
initialGraph.addEdge(2, 3)
for i in range(5):
    initialGraph.addEdge(i,i)
    k=2
    generator = KroneckerGenerator(initialGraph, k)
    graph = generator.generate()