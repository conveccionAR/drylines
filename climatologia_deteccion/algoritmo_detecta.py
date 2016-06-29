# Cargamos las cosas necesarias para correr:
#-*- coding: utf-8 -*- # Con esto no jode al escribir acentos
from __future__ import print_function # make sure print behaves the same in 2.7 and 3.x

from netCDF4 import Dataset, num2date, date2index      # Librerías netCDF
import netCDF4
import numpy as np   # Numerical python
import datetime
from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt
# Cosas para la parte que guarda en netcdf. Si supiera que hacen lo pondría.
import sys

#######################################################################################################
#######################################################################################################
# Este espacio queda reservado para definir parámetros que puedan modificarse para afinar el algoritmo.
# Dominio en el que hacemos el recorte:
lat_limites = [-46.0,-30.0]
lon_limites_normal = [-71.0, -56.0]
# Resto 360 para el formato de lats, lon, era interim
lon_limites = [360.0+lon_limites_normal[0], 360.0+lon_limites_normal[1]]

# Qué horas voy a analizar?? en principio veamos que pasa con los campos de las 18z, en donde las líneas deberían estar bien
# desarrolladas y la convección de la tarde todavía no modificó demasiado las condiciones. 
horas_dryline=['18']

# Cuál es el gradiente mínimo de q en el nivel más bajo del modelo que define una línea seca? en g/Kg cada 100 km.
grad_q_minimo = 3

# Cuál es el máximo gradiente de temperatura en el nivel más bajo del modelo permitido para diferenciar las líneas secas de un frente? en K/100Km.
grad_temp_max = 5 


#######################################################################################################
#######################################################################################################

# Defino todas las constantes que voy a usar y las funciones que voy a necesitar
const = {'pi':3.14159, 'dtr':3.14159/180, 'a':6.37122e6}


# La siguiente función calcula la derivada en la dirección "x" en coord. esféricas de un campo escalar 2D, dadas las latitudes y longitudes
def d_dx(matriz,lat,lon):
    # El dx cambia de acuerdo a la latitud, necesito un vector de deltas x
    dx0=np.diff(lon[0:2])*const['dtr']*const['a'] ; deltax=dx0[0]*np.cos(lat*const['dtr'])
    # Inicializo la variables con un array 2D del mismo tamaño que la matriz de entrada
    dmatrizdx = np.empty_like(matriz) ; dmatrizdx[:]=np.nan
    # Uso diferencias finitas para calcular la derivada direccional en una u otra dirección
    # Como van a ser centradas me tienen que quedar afuera los extremos, ojo con eso
    for i in range(len(lat)):
        dx = deltax[i]
        for j in range(1,len(lon)-1):
            dmatrizdx[i,j] = (matriz[i,j+1]-matriz[i,j-1]) / (2.0*dx)
    return dmatrizdx

# La siguiente función calcula la derivada en la dirección "y" en coord. esféricas de un campo escalar 2D, dadas las latitudes y longitudes
def d_dy(matriz,lat,lon):
    #El dy es constante, por lo que un sólo valor me alcanza.
    dy = np.diff(lat[0:2])*const['dtr']*const['a']
    # Inicializo la variable con un array 2D del mismo tamaño que la matriz de entrada
    dmatrizdy = np.empty_like(matriz) ; dmatrizdy[:]=np.nan
    # Uso diferencias finitas para calcular la derivada direccional
    # Como van a ser centradas me tienen que quedar afuera los extremos, ojo con eso
    for j in range(len(lon)):
        for i in range(1,len(lat)-1):
            dmatrizdy[i,j] = (matriz[i+1,j]-matriz[i-1,j]) / (2.0*dy)
    return dmatrizdy

# La siguiente función calcula la magnitud del gradiente en coord. esféricas de un campo escalar 2D, dadas las latitudes y longitudes
def grad_esfer(matriz,lat,lon):
    # Inicializo las variables con un array 2D dl mismo tamaño que la matriz de entrada
    dmdx = np.empty_like(matriz) ; dmdx[:]=np.nan
    dmdy = np.empty_like(matriz) ; dmdy[:]=np.nan
    # Calculo las derivadas direccionales usando las funciones d_dx y d_dy
    dmdx = d_dx(matriz, lat, lon); dmdy = d_dy(matriz,lat,lon)
    # Tengo las derivadas direccionales, busco la magnitud del gradiente, primero la inicializo del mismo tamaño
    grad_matriz = np.empty_like(matriz); grad_matriz[:] = np.nan        
    grad_matriz = np.sqrt((dmdx*dmdx)+(dmdy*dmdy))
    return grad_matriz

# Primer paso: abrir los datos y recortar el área de estudio
# Definimos un objeto f (file) que representa nuestro archivo netCDF y uno g con la máscara para el océano.
f = netCDF4.Dataset('/media/hernymet/datos/home/reanalisis/ERA-Int_modlev_20160101.nc','r')
g = netCDF4.Dataset('/media/hernymet/datos/home/reanalisis/land_sea_mask_05.nc','r')  # Land sea mask
# Accedemos a las variables, primero las dimensiones. Definimos objetos que se refieran a esas variables,
# después extraemos los valores. Hacerlo así me permite ver los atributos de cada una cuando haga falta.
times = f.variables['time']        # Calendario gregoriano, horas desde 1900-01-01 00:00:0.0
lat = f.variables['latitude']
lon = f.variables['longitude']
lsmask = g.variables['lsm']
# Ahora si los valores
time = times[:]; lats = lat[:]; lons = lon[:];lsm = lsmask[:]

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
temperatura = f.variables['t']; hum_esp = f.variables['q'];

# Y ahora si accedemos a las variables recortadas. COMO EN ESTA PRUEBA SÓLO BAJÉ EL NIVEL MÁS BAJAO las 
# dimensiones son tiempo, latitud, longitud.
# Recordar que en python el primer índice es inclusivo y el segundo exclusivo (de ahí el +1)
temp = temperatura[indices_horas,ind_lat_min:ind_lat_max+1, ind_lon_min:ind_lon_max+1]
q = hum_esp[indices_horas,ind_lat_min:ind_lat_max+1, ind_lon_min:ind_lon_max+1]
land_sea_mask = lsm [0,ind_lat_min:ind_lat_max+1, ind_lon_min:ind_lon_max+1]

# Recortamos las variables de las dimensiones también
lats_1 = lat[ind_lat_min:ind_lat_max+1]
lons_1 = lon[ind_lon_min:ind_lon_max+1]-360
times_1 = time[indices_horas]

###########################################
# Comienza el algoritmo propiamente dicho #
###########################################

# Voy a definir arrays "flags_condición" con las mismas dimensiones que mis variables recortadas. Van a tener unos si se cumplen las condiciones de línea seca
# o ceros si no. Cada condición va a tener un array de flags, y luego los voy a sumar todos a un array de "flags" general.
# Si se tienen que cumplir 4 condiciones, sólo los elementos de este array que valgan 4 son en dónde voy a buscar una línea seca.

#####################
# Primera condición #
#####################

# Calculo el gradiente de q en el nivel mas bajo para todos los tiempos, y descarto valores menores al umbral elegido al inicio del programa.

# Inicializo la variable gradiente de q con el mismo tamaño que q
grad_q = np.empty_like(q); grad_q[:] = np.nan
# Ahora si llamo a la función grad_esfer para el cálculo para todos los tiempos disponibles:
for i in range(q.shape[0]):
    grad_q[i,:,:] = grad_esfer(q[i,:,:], lats_1, lons_1)

# Reemplazamos los nan's por -9999.9 para que no molesten
grad_q[np.isnan(grad_q)] = -9999.9
# Si el gradiente de q es mayor al umbral definido al inicio le asigno un uno, sino va cero. Multiplico por 1e8 para que queden (g/Kg)/100Km.
# Genero las "flags" asociadas a esa condición:
flags_grad_q = np.where(grad_q*1e8>grad_q_minimo , 1.0, 0.0)

# Defino el array flags (que unirá todas las condiciones). Como es la primera condición es exactamente igual a este.
flags = flags_grad_q

#####################
# Segunda condición #
#####################

# Se aplica sobre el gradiente de temperatura. Si el gradiente de temperatura es mayor al umbral definido al principio,
# podría estar relacionado con un frente, por lo que descarto esos puntos. Tiene en cuenta también que si la temperatura 
# apunta hacia el sur no debería ser un frente.

# Inicializo la variables gradiente de temperatura y derivada en y de la temperatura con el mismo tamaño que temp:
grad_temp = np.empty_like(temp); grad_temp [:] = np.nan
dtemp_dy = np.empty_like(temp); dtemp_dy [:] = np.nan

# Llamo a las funciones grad_esfer y d_dy para calcularlas para todos los tiempos disponibles:
for i in range(temp.shape[0]):
    grad_temp[i,:,:] = grad_esfer(temp[i,:,:], lats_1, lons_1)
    dtemp_dy[i,:,:] = d_dy(temp[i,:,:], lats_1, lons_1)

# Reemplazamos los nan's por -9999.9 para que no molesten
grad_temp[np.isnan(grad_temp)] = -9999.9
dtemp_dy[np.isnan(grad_temp)] = -9999.9

# Finalmente generamos las "flags" asociadas a esta condición. Si el gradiente es mayor al umbral y la temperatura aumenta 
# hacia el norte lo descarto como posible frente (no suma flag).
# Inicializo el array de flags asociado a esta condición
flags_grad_temp = np.zeros_like(grad_temp)
# Y chequeo la condición para todos los tiempos (i), latitudes (j) y longitudes (k). Multiplico por 1e5 para pasar a K/100Km
for i in range(flags_grad_temp.shape[0]):
    for j in range(flags_grad_temp.shape[1]):
        for k in range(flags_grad_temp.shape[2]):
            if grad_temp[i,j,k]*1e5 > grad_temp_max and dtemp_dy[i,j,k] >0:
                flags_grad_temp[i,j,k] = 0
            else:
                flags_grad_temp[i,j,k] = 1

# Listo, sumo las flags de esta condición
flags = flags + flags_grad_temp


#####################
# Tercera condición #
#####################

# Enmascaro el océano.
for i in range(flags.shape[0]):
    flags[i,:,:] = flags[i,:,:] + land_sea_mask

# Guardamos las flags y todo lo que calculamos en un netcdf para poder abrir en GrADS
try: newdata.close()  # por seguridad, nos aseguramos que el archivo no esté abierto ya 
except: pass

newfile='/home/hernymet/Dropbox/drylines/climatologia_deteccion/flags_modlev.nc'
newdata=netCDF4.Dataset(newfile,'w')
newdata.title="flags_detección lineas secas"
newdata.Conventions="COARDS"
newdata.dataType='Grid'
newdata.history='SDF Netcdf con los cálculos para la detección de líneas secas'

newdata.createDimension('lat', len(lats_1))
newdata.createDimension('lon', len(lons_1))
newdata.createDimension('lev', 1)
newdata.createDimension('time', None)

### Create dimensional variables: ###
### For dimension variables, COARDS attributs are required for GrADS self describing
### Do latitude (f8 = float64):
latvar=newdata.createVariable('lat','f8',('lat'))
latvar.grads_dim='y'
latvar.grads_mapping='linear'
latvar.grads_size=str(len(lats_1))
latvar.units='degrees_north'
latvar.long_name='latitude'
latvar[:]=lats_1[:]

### Do longitude (f8 = float64):
lonvar=newdata.createVariable('lon','f8',('lon'))
lonvar.grads_dim='x'
lonvar.grads_mapping='linear'
lonvar.grads_size=str(len(lons_1))
lonvar.units='degrees_east'
lonvar.long_name='longitude'
lonvar[:]=lons_1[:]+360           # Vuelvo a sumar 360 para que no joda el grads

### Do levels (f8 = float64):
### removed grads_size from attributes
levvar=newdata.createVariable('lev','f8',('lev'))
levvar.grads_dim='z'
levvar.grads_mapping='levels'
levvar.units='millibar'
levvar.long_name='altitude'
# lonvar.grads_size=str(len(levs))
# levvar.minimum='1000.'
# levvar.maximum='10.'
# levvar.resolution='39.6'
# levvar[:]=levs[:]
levvar[:]=0                       # En principio tengo un solo nivel que voy a tomar como cero.

### Do Time (f8 = float64):
### Added grads_min and grads_step to attributes
timevar=newdata.createVariable('time','f8',('time'))
timevar.grads_dim='t'
timevar.grads_mapping='linear'
timevar.grads_size=str(8)
timevar.units='hours since 1900-01-01 00:00:0.0'
timevar.grads_step='24hr'
timevar.long_name='time'
timevar.minimum='18z01JAN2016'
timevar.maximum='18z31JAN2016'
timevar[:]=times_1[:]

# Ahora guardo todas las variables y flags que calculé
### There is a little more flexibility with this variables attributes!
### (f4=float32) ###
flagsvar=newdata.createVariable('flags','f4',('time','lat','lon'),fill_value=-9999.9)
flagsvar.long_name='flags'
flagsvar.units='adimensional'
flagsvar[:]=flags[:,:,:]

flags_grad_q_var=newdata.createVariable('flags_grad_q','f4',('time','lat','lon'),fill_value=-9999.9)
flags_grad_q_var.long_name='flags_grad_q'
flags_grad_q_var.units='adimensional'
flags_grad_q_var[:]=flags_grad_q[:,:,:]

flags_grad_temp_var=newdata.createVariable('flags_grad_temp','f4',('time','lat','lon'),fill_value=-9999.9)
flags_grad_temp_var.long_name='flags_grad_temp'
flags_grad_temp_var.units='adimensional'
flags_grad_temp_var[:]=flags_grad_temp[:,:,:]

flags_lsm_var=newdata.createVariable('flags_lsm','f4',('lat','lon'),fill_value=-9999.9)
flags_lsm_var.long_name='flags_lsm'
flags_lsm_var.units='adimensional'
flags_lsm_var[:]=land_sea_mask[:,:]

temp_var=newdata.createVariable('temp','f4',('time','lat','lon'),fill_value=-9999.9)
temp_var.long_name='temp'
temp_var.units='K'
temp_var[:]=temp[:,:,:]

grad_temp_var=newdata.createVariable('grad_temp','f4',('time','lat','lon'),fill_value=-9999.9*1e5)
grad_temp_var.long_name='grad_temp'
grad_temp_var.units='K/100Km'
grad_temp_var[:]=grad_temp[:,:,:]*1e5

dtemp_dyvar=newdata.createVariable('dt_dy','f4',('time','lat','lon'),fill_value=-9999.9*1e5)
dtemp_dyvar.long_name='dt_dy'
dtemp_dyvar.units='K/100km'
dtemp_dyvar[:]=dtemp_dy[:,:,:]*1e5

q_var=newdata.createVariable('q','f4',('time','lat','lon'),fill_value=-9999.9)
q_var.long_name='q'
q_var.units='Kg/Kg'
q_var[:]=q[:,:,:]

grad_q_var=newdata.createVariable('grad_q','f4',('time','lat','lon'),fill_value=-9999.9*1e8)
grad_q_var.long_name='grad_q'
grad_q_var.units='g/Kg/100Km'
grad_q_var[:]=grad_q[:,:,:]*1e8


newdata.close()
