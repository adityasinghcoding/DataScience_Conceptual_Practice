# Numpy Documentation Practice

### Freestyle warmup


```python
import numpy as np
```


```python
arr = [['a', 'd', 'i', 't', 'y', 'a'],
       ['s','i','n','g','h'],
       [1,2,3,4,5,6,7,8,9,0],
      ['@',3]]
```


```python
type(arr)
```




    list




```python
arr[3][0]
```




    '@'




```python
dic = {'Name':'Aditya','Surname':'Singh'}
```


```python
type(dic)
```




    dict




```python
dic['Surname'] = 'Thakur'
```


```python
print(dic)
```

    {'Name': 'Aditya', 'Surname': 'Thakur'}
    


```python
tup = ('a', 'd', 'i', 't', 'y', 'a', 1,2,3,4,5,6,7,8,9,0)
```


```python
type(tup)
```




    tuple




```python
tup[6]
```




    1




```python
sets = 
```

# Numpy Practice

### By Aditya Singh



```python
a = np.array([[2,4,6,8],
             [10,12,14,16],
            [18,19,20,22]])
```


```python
print(a.shape)
print("\n",a,"\n")
a[1][2]
```

    (3, 4)
    
     [[ 2  4  6  8]
     [10 12 14 16]
     [18 19 20 22]] 
    
    




    14




```python
print(a[0:2,2])
a[1,3]
```

    [ 6 14]
    




    16




```python
b = np.array([1,2,3,4,5,6,7,8,9,0])
b
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])




```python
print(a*2)
a.ndim
```

    [[ 4  8 12 16]
     [20 24 28 32]
     [36 38 40 44]]
    




    2




```python
print(len(a.shape))
print(a.size)
print(a.ndim)
a.dtype
```

    2
    12
    2
    




    dtype('int32')




```python
import math
```


```python
a.size == math.prod(a.shape)
```




    True




```python
digits_array = np.ones(3)
digits_array
```




    array([1., 1., 1.])




```python
np.empty(2)
```




    array([ 2., 10.])




```python
print(np.arange(10))
print(np.arange(2,21,2))
print(np.linspace(1,10,num=6))
```

    [0 1 2 3 4 5 6 7 8 9]
    [ 2  4  6  8 10 12 14 16 18 20]
    [ 1.   2.8  4.6  6.4  8.2 10. ]
    


```python
x = np.ones(6, dtype=np.int64)
x
```




    array([1, 1, 1, 1, 1, 1], dtype=int64)




```python
array = np.arange(1,11)
array = sorted(array, reverse=True)
array
```




    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]




```python
import random
np.random.shuffle(array) or random.shuffle(array)
print(array)
```

    [3, 5, 8, 4, 6, 10, 7, 1, 9, 2]
    


```python
ran_array = np.array([9, 7, 1, 8, 10, 6, 4, 2, 3, 5])
np.sort(ran_array)
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])




```python
a = np.array([1,2,4,5,23])
b = np.array([32,2,5,2,3])

np.concatenate((a,b))
```




    array([ 1,  2,  4,  5, 23, 32,  2,  5,  2,  3])




```python
x = np.array([[1,2,4,5,23],[2,3,1,6,4]])
y = np.array([[32,2,5,2,3],[4,2,23,1,2]])

np.concatenate((x,y), axis=1)
```




    array([[ 1,  2,  4,  5, 23, 32,  2,  5,  2,  3],
           [ 2,  3,  1,  6,  4,  4,  2, 23,  1,  2]])




```python
arr_nd = np.array([[[1,2,3,4,5],
                    [6,7,8,9,0]],
                   
                   [[11,12,13,14,15],
                    [16,17,18,19,20]],
                   
                   [[21,22,23,24,25],
                   [26,27,28,29,30]],
                   
                   [[11,12,13,14,15],
                    [16,17,18,19,20]],

                   [[21,22,23,24,25],
                   [26,27,28,29,30]],

                   [[11,12,13,14,15],
                    [16,17,18,19,20]],
                  ])

print(arr_nd.ndim)
print(arr_nd.size)
arr_nd.shape
```

    3
    60
    




    (6, 2, 5)




```python
a = np.array([1,2,3,4,5,6])
b = a.reshape(2,3)
print(a)
b
```

    [1 2 3 4 5 6]
    




    array([[1, 2, 3],
           [4, 5, 6]])




```python
print(a.shape)
a2 = a[:, np.newaxis]
# a2 = a[np.newaxis, :]
print(a2.shape)
a2
```

    (6,)
    (6, 1)
    




    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])




```python
print(arr_nd)

print("="*60)
print(arr_nd[arr_nd>10])
condition = arr_nd[(arr_nd>10) | (arr_nd<20)]

print("="*60)
print(condition)

print("="*60)
print(arr_nd[arr_nd%2==0])

print("="*60)
above10 = (arr_nd > 20) | (arr_nd==20)
print(above10)
```

    [[[ 1  2  3  4  5]
      [ 6  7  8  9  0]]
    
     [[11 12 13 14 15]
      [16 17 18 19 20]]
    
     [[21 22 23 24 25]
      [26 27 28 29 30]]
    
     [[11 12 13 14 15]
      [16 17 18 19 20]]
    
     [[21 22 23 24 25]
      [26 27 28 29 30]]
    
     [[11 12 13 14 15]
      [16 17 18 19 20]]]
    ============================================================
    [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 11 12 13 14
     15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 11 12 13 14 15 16 17 18
     19 20]
    ============================================================
    [ 1  2  3  4  5  6  7  8  9  0 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
     29 30 11 12 13 14 15 16 17 18 19 20]
    ============================================================
    [ 2  4  6  8  0 12 14 16 18 20 22 24 26 28 30 12 14 16 18 20 22 24 26 28
     30 12 14 16 18 20]
    ============================================================
    [[[False False False False False]
      [False False False False False]]
    
     [[False False False False False]
      [False False False False  True]]
    
     [[ True  True  True  True  True]
      [ True  True  True  True  True]]
    
     [[False False False False False]
      [False False False False  True]]
    
     [[ True  True  True  True  True]
      [ True  True  True  True  True]]
    
     [[False False False False False]
      [False False False False  True]]]
    


```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
np.nonzero(a>7)
```




    (array([1, 2, 2, 2, 2], dtype=int64), array([3, 0, 1, 2, 3], dtype=int64))




```python
a2 = a[1:2,1:3]
a2
```




    array([[6, 7]])




```python
x = np.array([[1,2],
              [2,3]])
y = np.array([[3,4],
              [5,1]])

vstack = np.vstack((x,y))
print(vstack)

print("="*60)
hstack = np.hstack((x,y))
print(hstack)
```

    [[1 2]
     [2 3]
     [3 4]
     [5 1]]
    ============================================================
    [[1 2 3 4]
     [2 3 5 1]]
    


```python
np.hsplit(hstack,2)
```




    [array([[1, 2],
            [2, 3]]),
     array([[3, 4],
            [5, 1]])]




```python
nx = np.arange(1,25).reshape(2,12)
nx
```




    array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
           [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])




```python
np.hsplit(nx,3)
```




    [array([[ 1,  2,  3,  4],
            [13, 14, 15, 16]]),
     array([[ 5,  6,  7,  8],
            [17, 18, 19, 20]]),
     array([[ 9, 10, 11, 12],
            [21, 22, 23, 24]])]




```python
temp = hstack[0,:]
temp[1] = temp[1]*100
hstack
```




    array([[  1, 200,   3,   4],
           [  2,   3,   5,   1]])




```python
ar = hstack.copy()
ar
```




    array([[  1, 200,   3,   4],
           [  2,   3,   5,   1]])




```python
ones = np.ones(8, dtype=int)
ones = ones.reshape(2,4)
print(ones)
```

    [[1 1 1 1]
     [1 1 1 1]]
    


```python
ar_ones = ar + ones
print(ar_ones)
print()
print(ar_ones - ones)
print()
print(ar_ones * ar_ones)
```

    [[  2 201   4   5]
     [  3   4   6   2]]
    
    [[  1 200   3   4]
     [  2   3   5   1]]
    
    [[    4 40401    16    25]
     [    9    16    36     4]]
    


```python
ar_ones = ar_ones.reshape(4,2)
print(vstack)
print("="*60)
print(ar_ones)
```

    [[1 2]
     [2 3]
     [3 4]
     [5 1]]
    ============================================================
    [[  2 201]
     [  4   5]
     [  3   4]
     [  6   2]]
    


```python
mul = vstack * ar_ones
print(mul)
```

    [[  2 402]
     [  8  15]
     [  9  16]
     [ 30   2]]
    


```python
total = mul.sum()
print(total)
print()
print(mul.sum(axis=0))
```

    484
    
    [ 49 435]
    


```python
print(mul / 2)
print("="*60)
print(mul)
```

    [[  1.  201. ]
     [  4.    7.5]
     [  4.5   8. ]
     [ 15.    1. ]]
    ============================================================
    [[  2 402]
     [  8  15]
     [  9  16]
     [ 30   2]]
    


```python
print(mul.max())
print(mul.min())
print(mul.sum())

print("="*60)

print(mul.max(axis=0))
print(mul.max(axis=1))

```

    402
    2
    484
    ============================================================
    [ 30 402]
    [402  15  16  30]
    


```python
mat = np.array([[2,3],[25,6],[3,5],[6,7]])
print(mat)
print("="*60)

print(mat[1:3])
print("="*60)

print(mat[1:3,:1])
print("="*60)

print(mat[0:2,1:])
print()
```

    [[ 2  3]
     [25  6]
     [ 3  5]
     [ 6  7]]
    ============================================================
    [[25  6]
     [ 3  5]]
    ============================================================
    [[25]
     [ 3]]
    ============================================================
    [[3]
     [6]]
    
    


```python
print(np.random.default_rng().random(2))
print("="*60)

rng = np.random.default_rng()
print(rng.random(3))
print("="*60)

print(np.random.random((3,5,2)))
print("="*60)

rng.integers(10, size=(6,2))
```

    [0.75950322 0.15490196]
    ============================================================
    [0.34468563 0.82803041 0.60726843]
    ============================================================
    [[[0.55042599 0.73073375]
      [0.52967851 0.42120456]
      [0.15438924 0.98178031]
      [0.94602613 0.26653204]
      [0.15510913 0.20160757]]
    
     [[0.6898786  0.29669514]
      [0.54854562 0.35226412]
      [0.2838485  0.06673392]
      [0.19325892 0.63286415]
      [0.84572164 0.28631838]]
    
     [[0.46374475 0.84816812]
      [0.57876391 0.55887519]
      [0.18570421 0.87989428]
      [0.9319787  0.95212956]
      [0.47165536 0.71674332]]]
    ============================================================
    




    array([[4, 4],
           [8, 3],
           [3, 0],
           [6, 3],
           [5, 2],
           [5, 6]], dtype=int64)




```python
rng.integers(100,size=(3,3))
```




    array([[33, 35,  3],
           [80,  1, 76],
           [13, 47, 81]], dtype=int64)




```python
ua = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
ua = ua.reshape(3,5)
print(ua)
print("="*60)

uaa = np.unique(ua).reshape(2,5)
print(uaa)
```

    [[11 11 12 13 14]
     [15 16 17 12 13]
     [11 14 18 19 20]]
    ============================================================
    [[11 12 13 14 15]
     [16 17 18 19 20]]
    


```python
uaa, indices = np.unique(ua, return_index=True)
print(indices)
print(uaa)

print("="*60)

uaa, occurence = np.unique(ua, return_counts=True)
print(occurence)
print(uaa)
print()
uaa = uaa.flatten()
occurence = occurence.flatten()
final = np.vstack((uaa, occurence))
# final = np.concatenate((uaa, occurence))
print(final.T)
```

    [ 0  2  3  4  5  6  7 12 13 14]
    [11 12 13 14 15 16 17 18 19 20]
    ============================================================
    [3 2 2 2 1 1 1 1 1 1]
    [11 12 13 14 15 16 17 18 19 20]
    
    [[11  3]
     [12  2]
     [13  2]
     [14  2]
     [15  1]
     [16  1]
     [17  1]
     [18  1]
     [19  1]
     [20  1]]
    


```python
a2d = np.array([[2,3,3],[2,5,3],[2,6,4]])
print(a2d)
print("="*60)
print(np.unique(a2d))
```

    [[2 3 3]
     [2 5 3]
     [2 6 4]]
    ============================================================
    [2 3 4 5 6]
    


```python
narr = np.arange(1,51)
narr = narr.reshape(5,10)
narr1, narr2 = np.hsplit(narr, 2)
print(narr1, "\n\n",narr2)
print("="*60)
print(narr1)
print("="*10,"vs","="*10)
print(narr1.T)
print("="*60)
print("Reversed/Flipped Matrix\n",np.flip(narr2))
print()
print("Reversed along Y axis\n",np.flip(narr2, axis = 1))
print()
print("Reversed the subset of narr2\n",np.flip(narr2[1:4,1:4]))
```

    [[ 1  2  3  4  5]
     [11 12 13 14 15]
     [21 22 23 24 25]
     [31 32 33 34 35]
     [41 42 43 44 45]] 
    
     [[ 6  7  8  9 10]
     [16 17 18 19 20]
     [26 27 28 29 30]
     [36 37 38 39 40]
     [46 47 48 49 50]]
    ============================================================
    [[ 1  2  3  4  5]
     [11 12 13 14 15]
     [21 22 23 24 25]
     [31 32 33 34 35]
     [41 42 43 44 45]]
    ========== vs ==========
    [[ 1 11 21 31 41]
     [ 2 12 22 32 42]
     [ 3 13 23 33 43]
     [ 4 14 24 34 44]
     [ 5 15 25 35 45]]
    ============================================================
    Reversed/Flipped Matrix
     [[50 49 48 47 46]
     [40 39 38 37 36]
     [30 29 28 27 26]
     [20 19 18 17 16]
     [10  9  8  7  6]]
    
    Reversed along Y axis
     [[10  9  8  7  6]
     [20 19 18 17 16]
     [30 29 28 27 26]
     [40 39 38 37 36]
     [50 49 48 47 46]]
    
    Reversed the subset of narr2
     [[39 38 37]
     [29 28 27]
     [19 18 17]]
    


```python
x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(x.flatten())
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12]
    


```python
np.save('Numpy_Doc_Practice.npy',a)
```


```python

```
