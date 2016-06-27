# Cargamos las cosas necesarias para correr:
#-*- coding: utf-8 -*- # Con esto no jode al escribir acentos
from netCDF4 import Dataset, num2date, date2index      # Librerías netCDF
import netCDF4
import numpy as np   # Numerical python
import datetime
from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt
#######################################################################################################
#######################################################################################################
# Este espacio queda reservado para definir parámetros que puedan modificarse para afinar el algoritmo.
# Dominio en el que hacemos el recorte:
lat_limites = [-46.0,-30.0]
lon_limites_normal = [-71.0, -56.0]
# Resto para el formato de lats, lon, era interim
lon_limites = [360.0+lon_limites_normal[0], 360.0+lon_limites_normal[1]]
# Qué horas voy a analizar?? en principio veamos que pasa con los campos de las 18z
horas_dryline=['18']

# Cuál es el gradiente mínimo de q en 925 hPa que define una línea seca? en g/Kg cada 100 km.
grad_q_minimo = 3

#######################################################################################################
#######################################################################################################

# Defino todas las constantes que voy a usar y las funciones que voy a necesitar
const = {'pi':3.14159, 'dtr':3.14159/180, 'a':6.37122e6}

# La siguiente función calcula la magnitud del gradiente en coord. esféricas de un campo escalar 2D, dadas las latitudes y longitudes
def grad_esfer(matriz,lat,lon):
    #El dy es constante, por lo que un sólo valor me alcanza. Tiene un menos adelante por el orden de las lats del ERA-interim
    dy = -np.diff(lat[0:2])*const['dtr']*const['a']
    # El dx cambia de acuerdo a la latitud, necesito un vector de deltas x
    dx0=np.diff(lon[0:2])*const['dtr']*const['a'] ; deltax=dx0[0]*np.cos(lat*const['dtr'])
    # Inicializo las variables con un array 2D dl mismo tamaño que la matriz de entrada
    dmatrizdx = np.empty_like(matriz) ; dmatrizdx[:]=np.nan
    dmatrizdy = np.empty_like(matriz) ; dmatrizdy[:]=np.nan
    # Uso diferencias finitas para calcular las derivadas direccionales en una u otra dirección
    # Como van a ser centradas me tienen que quedar afuera los extremos, ojo con eso
    # Derivada en dirección x
    for i in range(len(lat)):
        dx = deltax[i]
        for j in range(1,len(lon)-1):
            dmatrizdx[i,j] = (matriz[i,j+1]-matriz[i,j-1]) / (2.0*dx)
    # Derivada en dirección y
    for j in range(len(lon)):
        for i in range(1,len(lat)-1):
            dmatrizdy[i,j] = (matriz[i+1,j]-matriz[i-1,j]) / (2.0*dy)
    # Tengo las derivadas direccionales, busco la magnitud del gradiente, primero la inicializo del mismo tamaño
    grad_matriz = np.empty_like(matriz); grad_matriz[:] = np.nan        
    grad_matriz = np.sqrt((dmatrizdx*dmatrizdx)+(dmatrizdy*dmatrizdy))
    return grad_matriz


# Primer paso: abrir los datos y recortar el área de estudio
# Definimos un objeto f (file) que representa nuestro archivo netCDF
f = netCDF4.Dataset('/media/hernymet/datos/home/reanalisis/ERA-Int_pl_20160101.nc','r')
# Accedemos a las variables, primero las dimensiones. Definimos objetos que se refieran a esas variables,
# después extraemos los valores. Hacerlo así me permite ver los atributos de cada una cuando haga falta.
times = f.variables['time']        # Calendario gregoriano, horas desde 1900-01-01 00:00:0.0
lat = f.variables['latitude']
lon = f.variables['longitude']
lev = f.variables['level']
# Ahora si los valores
time = times[:]; lats = lat[:]; lons = lon[:]; levs = lev[:]

# Al abrir las demás variables ya me quedo sólo con el recorte definido al principio
# Necesito extraer los índices de los vectores lats y lons más cercanos a los que definí en mi región de estudio
# Por alguna razón las latitudes están en orden decreciente y las longitudes en orden creciente
ind_lat_min = np.argmin( np.abs( lats - lat_limites[1] ) )
ind_lat_max = np.argmin( np.abs( lats - lat_limites[0] ) ) 
ind_lon_min = np.argmin( np.abs( lons - lon_limites[0] ) )
ind_lon_max = np.argmin( np.abs( lons - lon_limites[1] ) )

# Pasamos los tiempos del calendario Gregoriano a un formato más conocido
fechas = num2date(time[:],'hours since 1900-01-01 00:00:0.0' )
#fechas=[date.strftime('%Y-%m-%d %H:%M:%S') for date in fechas[:]]   # Esto lo dejo comentado, puede servir después.

# Extraemos un vector sólo con las horas, así podemos filtrar sólo las que elegimos
horas=[date.strftime('%H') for date in fechas[:]]
# Buscamos los índices que coincidan con las "horas_dryline" definidas al principio
indices_horas = [i for i, x in enumerate(horas) if x == horas_dryline[0]]

# Defino objetos referidos a las demás variables
geopotencial = f.variables['z']; temperatura = f.variables['t']; hum_esp = f.variables['q']; vvel = f.variables['w']
v_zonal = f.variables['u']; v_meridio = f.variables['v']
# Y ahora si accedemos a las variables recortadas. Las dimensiones son tiempo, nivel, latitud, longitud.
# Recordar que en python el primer índice es inclusivo y el segundo exclusivo (de ahí el +1)
geop = geopotencial[indices_horas,:,ind_lat_min:ind_lat_max+1, ind_lon_min:ind_lon_max+1]
temp = temperatura[indices_horas,:,ind_lat_min:ind_lat_max+1, ind_lon_min:ind_lon_max+1]
q = hum_esp[indices_horas,:,ind_lat_min:ind_lat_max+1, ind_lon_min:ind_lon_max+1]
omega = vvel[indices_horas,:,ind_lat_min:ind_lat_max+1, ind_lon_min:ind_lon_max+1]
u = v_zonal[indices_horas,:,ind_lat_min:ind_lat_max+1, ind_lon_min:ind_lon_max+1]
v = v_meridio[indices_horas,:,ind_lat_min:ind_lat_max+1, ind_lon_min:ind_lon_max+1]
# Recortamos las variables de las dimensiones también
lats_1 = lat[ind_lat_min:ind_lat_max+1]
lons_1 = lon[ind_lon_min:ind_lon_max+1]-360
times_1 = time[indices_horas]


###########################################
# Comienza el algoritmo propiamente dicho #
###########################################

# Defino un array "flags" con las mismas dimensiones que mis variables recortadas, que va a tener valores si se cumplen las condiciones de línea seca
# o ceros si no. Los valores van a ser un número entero que dependerá de cuántas condiciones use. Si se tienen que cumplir 4 condiciones, sólo los
# elementos de flags que valgan 4 son en dónde voy a buscar una línea seca.

flags = np.zeros_like(q);


#####################
# Primera condición #
#####################

# Calculo el gradiente de q en 925 para todos los tiempos, e intento generar un array que sirva como una máscara para descartar valores menores al
# umbral elegido al inicio del programa.
# Genero un array con la humedad específica en el nivel de 925 hPa
ind_925 = [i for i, l in enumerate(levs) if l ==925]
q_925 = q[:,ind_925,:,:]
# Inicializo la variable gradiente de q en 925 con el mismo tamaño que q_925
grad_q_925 = np.empty_like(q_925); grad_q_925[:] = np.nan
# Ahora si llamo a la función grad_esfer para el cálculo para todos los tiempos disponibles:
for i in range(q_925.shape[0]):
    grad_q_925[i,:,:,:] = grad_esfer(np.squeeze(q_925[i,:,:,:]), lats_1, lons_1)

# Reemplazamos los nan's por -9999.9 para que no molesten
grad_q_925[np.isnan(grad_q_925)] = -9999.9
# si el gradiente de q es mayor al umbral definido al inicio le asigno un uno, sino va cero, genero las "flags" asociadas a esa condición:
flags_grad_q_925 = np.where(grad_q_925*1e8>grad_q_minimo , 1.0, 0.0)

# Se lo sumo al array flags (que unirá todas las condiciones) elemento por elemento para incorporar esta condición
flags = flags+flags_grad_q_925

#####################
# Segunda condición #
#####################






















