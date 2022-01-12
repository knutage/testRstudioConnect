# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:29:29 2020

@author: oslkme
"""

import pandas as pd
import numpy as np
import streamlit as st
#import ODS_connect

np.random.seed(24)
df = pd.DataFrame({' ': np.array(['Ca', 'Al', 'Fe', 'Si', 'S', 'K', 'Cr', 'Mg', 'Ti', 'Na'])})
df = pd.concat([df, pd.DataFrame(np.random.randn(10, 7), columns=['27.05', '26.05','25.05','24.05','23.05','22.05','21.05'])],
               axis=1)


#newdf1 = pd.DataFrame([['Ovn 1','Ovn2', 'Ovn2', '968', '940', '968', '955', 'TMS' ]], columns = df.columns)
#newdf2 = pd.DataFrame([['Ovn 2','955', '955', '968', '968', '968', '940', '940' ]], columns = df.columns)
#newdf3 = pd.DataFrame([['Ovn 3','TMS', 'TMS', 'TMS', '940', '955', '955', '955' ]], columns = df.columns)
#df = df.astype('str')
#df = df.append(newdf1,  ignore_index = True)
#df = df.append(newdf2,  ignore_index = True)
#df = df.append(newdf3,  ignore_index = True)

df.iloc[3, 3] = np.nan
df.iloc[0, 2] = np.nan

df = df.set_index(' ')


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if isinstance(val, str):
        color = 'black'
    else: 
        color = 'red' if val < 0 else 'black'
    return 'color: %s' % color


def background_color(val):
    if isinstance(val, str):
        if val is '968':
            color = '#7bbad8'
        elif val is 'TMS':
            color = '#6ba7c3'
        elif val is '955':
            color = '#568398'
        elif val is '940':
            color = '#446778'
        else:
            color = 'white'
    else: 
        color= 'white'
    return 'background-color: %s' % color

s = df.style.applymap(color_negative_red)
s = s.applymap(background_color)



# Bygger Streamlit GUI:
"""
# Dette er en enkel demo i Streamlit 
## Forklaring: 


Her kan du legge inn den teksten du ønsker *og formatere* **den**.  
Husk 2 mellomrom for linjeskift  
 ___ 
blablablabla .... ....  
 --- 
- punkt1
- punkt2
"""

st.button('Oppdater data')

st.table(s)
#
st.dataframe(s)
st.sidebar.markdown("forklaring ...")
tall = st.sidebar.number_input('test')
tekst = st.sidebar.text_input('test')
dato = st.sidebar.date_input('test')
tall2 = st.sidebar.slider('test')
#var = st.sidebar.radio('test')
st.write(tekst)


# part2:
import stumpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# generate signal
noSignal = pd.DataFrame(np.random.randn(100, 1))
signal = pd.DataFrame(np.sin(np.arange(0., 360.,10.) * np.pi / 180. ))

df = pd.DataFrame()
for i in range(10):
    df = df.append(noSignal, ignore_index= True)
    df = df.append(signal, ignore_index= True)

noise = np.random.randn(len(df), 1)*0.1
df = df+ noise # add noise


# detection parameters
m = int(36)
patternIdX = 100


#run detetection
mp = stumpy.stump(df.iloc[:,0], m)
selectedPattern = df.iloc[:,0][patternIdX:patternIdX+m]
distance_profile = stumpy.core.mass(selectedPattern, df.iloc[:,0])
idx = np.argmin(distance_profile) 
# print(idx)

# select # of outputs
k = 10
idxN = np.argpartition(distance_profile, k)[:(k)] 
# print(idxN)

# plot results
plt.figure()
df.iloc[:,0].plot(color = 'grey') # data
for j in idxN: 
    df.iloc[:,0][j:j+m].plot() # detected instances
selectedPattern.plot(color = 'black') #orginal pattern
