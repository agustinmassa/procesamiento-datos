"""
PROCESAMIENTO DE DATOS EN EL DATASET arbolado-publico-lineal-2017-2018.csv

a) Cargando los datos. Importen este nuevo dataset usando pandas. 
Van a notar que les da una advertencia (warning) porque hay algunas columnas con tipos mezclados. 
Por ahora ignorenlo.Para ahorrarnos trabajo, definan un nuevo DataFrame usando solo las columnas
['nro_registro', 'nombre_cientifico', 'estado_plantera', 'ubicacion_plantera', 'nivel_plantera', 'diametro_altura_pecho', 'altura_arbol'].


b) Limpieza de datos (I). Analicen los valores únicos que pueden tomar las columnas 
'estado_plantera', 'ubicacion_plantera' y 'nivel_plantera'. ¿Qué es lo que ven?
Para las tres columnas, unifiquen los valores que pertecen a una misma catgoría.


c) Limpieza de datos (II). Hagan histogramas de los valores de las variables 'diametro_altura_pecho' y 'altura_arbol'.
A primera vista no parece haber nada raro, pero fijense que para el diámetro (que está medido en cm) hay muchos datos con valor 0 (pueden usar el método value_counts()).
Si bien podría haber árboles con menos de 1 cm de diámetro, la cantidad de los mismos nos hace sospechar que en gran parte de los casos se trata de un error.
Eliminen las filas con diámetro 0, o al menos por ahora reemplacen el valor por nan.

d) Datos faltantes. Analicen la cantidad de datos faltantes en cada columna y decidan qué hacer con ellos 
(descartarlos, crear una nueva categoría en las variables categóricas, reemplazarla por promedio/mediana en las numéricas, etc.)

e) Variables categóricas. Apliquen el método de One-Hot Encoding a alguna de las variables categóricas del dataset. 
¿De qué va a depender la cantidad de componentes de los vectores resultantes?

"""
#A)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



ARCHIVO = 'F:\Agu\Diplomatura.UNSAM\Practica\Procesamiento de datos\Arbolado-publico-lineal-2017-2018.csv'
arboles = pd.read_csv(ARCHIVO)

columnas = ['nro_registro', 'nombre_cientifico', 'estado_plantera', 'ubicacion_plantera', 'nivel_plantera', 'diametro_altura_pecho', 'altura_arbol']
arboles = arboles[columnas] #Elijo solo las columnas

#print(arboles.info())

#B)

#ESTADO
print(arboles.estado_plantera.unique())
#Hago un diccionario para indicar que valores hay que remplazar
dic_estado  = {'ocupada':'Ocupada', 'Ocupada ':'Ocupada', 
              'SobreOcupada':'Sobreocupada', 'sobreocupada':'Sobreocupada'}

for clave, valor in dic_estado.items():
    arboles.estado_plantera.replace(clave, valor, inplace=True)
print(arboles.estado_plantera.unique())

#UBICACION
print(arboles.ubicacion_plantera.unique())
dic_ubicacion = {'regular':'Regular', 'Regular ':'Regular','o':'O',  
              'Och':'Ochava', 'ochava':'Ochava', 'Ochva':'Ochava',
              'Fuera Línea,Ochava':'Ochava/Fuera Línea',
              'Fuera de Línea, Ochava':'Ochava/Fuera Línea',
              'Fuera Línea/Ochava':'Ochava/Fuera Línea'}

for clave, valor in dic_ubicacion.items():
    arboles.ubicacion_plantera.replace(clave, valor, inplace=True)
print(arboles.ubicacion_plantera.unique())

#NIVEL
print(arboles.nivel_plantera.unique())
dic_nivel = {'A  Nivel':'A Nivel', 'a Nivel':'A Nivel', 'A nivel':'A Nivel', 
             'A nivel ':'A Nivel', 'An':'A Nivel', 'AN':'A Nivel',
             'Baja Nivel':'Bajo Nivel', 'Bajo  nivel':'Bajo Nivel', 
             'Bajo Bivel':'Bajo Nivel', 'bajo nivel':'Bajo Nivel', 
             'Bajo nivel':'Bajo Nivel', 'BN':'Bajo Nivel', 'el':'Elevada', 
             'EL':'Elevada', 'elevada':'Elevada', 'ELEVADA':'Elevada', 
             'Elevadas':'Elevada', 'Elevado':'Elevada', 'Eleveda':'Elevada'}

for clave, valor in dic_nivel.items():
    arboles.nivel_plantera.replace(clave, valor, inplace=True)
print(arboles.nivel_plantera.unique())

print('Hay algunas categorias que probablemente sean erroneas y podrian ser descartadas como por ejemplo ( obs: no tiene plantera definida)')

#C)

df = arboles
columns = ['diametro_altura_pecho', 'altura_arbol']
N_col = 2
N_rows = 1

fig, ax = plt.subplots(N_rows, N_col, figsize=(5*N_col,5*N_rows))

for i in range(N_col):
    ax[i].hist(df[columns[i*N_rows]], bins=70)
    ax[i].set_title(columns[i*N_rows])
#plt.show()

#Veo cuantos arboles con 0 hay
print('N arboles diametro 0:', arboles.diametro_altura_pecho.value_counts()[0.])

#Los cambio por nan
arboles.diametro_altura_pecho.replace(0.,np.nan, inplace=True)
print(arboles.diametro_altura_pecho.head())

#D)

print(arboles.isna().sum())
#Con un dataset tan grande, perder unos poco miles no es problema
arboles_limpio = arboles.dropna()
print(print(arboles_limpio.isna().sum()))
#Tambien se puede imputar con el SimpleImputer de Sklearn
#hay que tranformar las variables categoricas en variables numericas.

#E)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
cat_encoder.fit(arboles_limpio[['ubicacion_plantera']]) #el OneHotEncoder espera recibir un dataframe, no una sola columna,
#por eso el doble paréntesis abajo
ubicacion_plantera_OHE = cat_encoder.transform(arboles_limpio[['ubicacion_plantera']]).toarray()
print(ubicacion_plantera_OHE)
print('La cantidad de componentes de los vectores depende de la cantidad de categorías.')