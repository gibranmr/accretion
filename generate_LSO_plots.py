#!/usr/bin/env python
# coding: utf-8

from accretion_functions import *
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, solve_ivp
import os
import glob
from scipy.integrate import quad
from scipy.optimize import root_scalar
import mesa_reader as mr
from mesa_reader import MesaData 
from scipy.interpolate import interp1d
import argparse
import time

# Registrar el inicio de la simulación
start_time = time.time()
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
mpl.rc('text', usetex=True)
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.unicode'] = True  # Uncomment this if needed for older matplotlib versions
mpl.rcParams.update({'font.size': 18})

# Constants
number_of_points = int(4e2) #How many rows do you want in your DataFrame
#Angular range in theta (theta=0 corresponds to the north pole and theta=pi/2 to the equator)
#For 1pi/6 < theta < 5pi/6 we treat a tours belt over a +/- 60º around the equator.
theta_initial = 0 * np.pi/6 
theta_final = 6 * np.pi/6 
avg_norm_factor, _ = quad(lambda theta: np.sin(theta), theta_initial, theta_final)


G = 6.67430e-8  # Gravitational constant in cgs units
M_sun = 1.989e33  # Solar mass in g
R_sun = 6.957e10  # Radio solar en cm
#M = 14 * M_sun #Star's mass
#R_star = 5.18e10 #Star's radius
#K = 4.44540243e+14 # # Polytropic constant K (K = N_n_value * G * R * M**(1/3) in cgs units
gamma = 5/3 # Adiabatic index
c = 2.99792458e10 #Speed of light in cgs

# Argumentos dinámicos
parser = argparse.ArgumentParser(description="Generar gráficos combinados para diferentes masas y eta_D.")
parser.add_argument("--mass", type=float, required=True, help="Masa en M_sun.")
parser.add_argument("--eta_d", type=float, required=True, help="Dutch scaling factor.")
args = parser.parse_args()

mass = args.mass
eta_d = args.eta_d

rotation = 0.9  # Fracción crítica de rotación
model = f'{int(mass)}M_W{rotation}_D{eta_d}'

# Ruta base a los perfiles de MESA (fija)
#base_path = '/fs/phaethon/other0/gibran/sim_results/Si28_5d-2_W9_D08/'
base_path = f'/fs/phaethon/other0/gibran/sim_results/Si28_5d-2_W9_D{eta_d:.1f}/'


# Ruta fija para guardar las gráficas
output_path = '/fs/phaethon/other0/gibran/accretion/plots'

# Crear directorio de salida si no existe
os.makedirs(output_path, exist_ok=True)


# Construir la ruta de logs
logs_dir = os.path.join(base_path, f'{int(mass)}M_W{rotation}_D{eta_d}')






# Función para extraer el número de perfil
def get_profile_number(filename):
    num_str = filename.split('profile')[-1].split('.data')[0]
    return int(num_str)

# Encuentra y ordena los archivos 'profile*.data'
profile_files = glob.glob(os.path.join(logs_dir, 'profile*.data'))
profile_files.sort(key=get_profile_number)

# Verificar si hay archivos disponibles
if not profile_files:
    raise FileNotFoundError(f"No se encontraron archivos 'profile*.data' en {logs_dir}")

# Cargar el último perfil
latest_profile = profile_files[-1]
profile_data = MesaData(latest_profile)

radius_values = profile_data.radius * R_sun  # Convert R_sun a cm
rho_values = 10**profile_data.logRho  # Densidad en g/cm^3 (logarithmic to linear)
Omega_values = profile_data.omega  # Omega (rad/s)

rho_interpolated = interp1d(radius_values, rho_values, kind='cubic', fill_value="extrapolate")
Omega_interpolated = interp1d(radius_values, Omega_values, kind='cubic', fill_value="extrapolate")

def M_r(r):
    result, _ = quad(lambda r_prime: 4 * np.pi * r_prime**2 * rho_interpolated(r_prime), 1e6, r)
    return result

# Create an r grid from 1e7 cm to the maximum radius in the profile
r_min = 1e7  # Starting radius in cm
r_max = np.max(profile_data.radius * R_sun)  # Convert from solar radii to cm
r_values = np.logspace(np.log10(r_min), np.log10(r_max), number_of_points)  # 100 grid points
data = []

for r in r_values:
    rho_val = rho_interpolated(r)
    Omega_r_val = Omega_interpolated(r)
    M_r_val = M_r(r)
    
    data.append([r, rho_val, M_r_val, Omega_r_val])

df = pd.DataFrame(data, columns=['r', 'rho_r', 'M_r', 'Omega_r'])

df['Omega_k'] = np.sqrt(G * df['M_r'] / df['r']**3)

#df['tau'] = np.log(df['rho_r']).diff() / np.log(df['r']).diff()

output_path = f'Data/data_{model}.csv'
df.to_csv(output_path, index=False, float_format='%.8e')
#print(f"Archivo guardado en: {output_path}")

df = pd.read_csv(f'Data/data_{model}.csv')


df = pd.read_csv(f'Data/data_{model}.csv')
from scipy.integrate import quad

def eccentricity(row, theta):
    return 1 - ((row['Omega_r'] / row['Omega_k']) * np.sin(theta))**2


def avg_teq(row):
    def t_eq_2v(theta):
        e = eccentricity(row, theta)
        parenthesis = (np.arccos(-e) + e * np.sqrt(1 - e**2)) * (1 + e)**(-1.5)
        #t_s_val = R_star / (np.sqrt(gamma * K * row['rho_r']**(2/3)))  
        t_s_val =  1 / row['Omega_k']
        t_eq_val = ( 1 / row['Omega_k']) * parenthesis + t_s_val
        return t_eq_val    
    integrand = lambda theta: t_eq_2v(theta) * np.sin(theta)
    integral, _ = quad(integrand,  theta_initial, theta_final)
    return integral / avg_norm_factor

def avg_req(row):
    def r_fb(theta):
        e = eccentricity(row, theta)
        return row['r'] * (1 - e)
    integrand = lambda theta: r_fb(theta) * np.sin(theta)
    integral, _ = quad(integrand, theta_initial, theta_final)
    avg_req_val = integral / avg_norm_factor
    return avg_req_val

def avg_ecc(row):
    integrand = lambda theta: eccentricity(row, theta) * np.sin(theta)
    integral, _ = quad(integrand, theta_initial, theta_final)
    return integral / avg_norm_factor

def avg_v_phi(row):
    def v_phi(theta):
        e = eccentricity(row, theta)
        return -(row['r'] * row['Omega_k'] * np.sin(theta)) / np.sqrt(1 - e)
    integral, _ = quad(v_phi, theta_initial, theta_final)
    return integral / avg_norm_factor



#df['c_s'] = np.sqrt(gamma * K * df['rho_r']**(2/3))
#df['t_s_polytropic'] = R_star / df['c_s']
#df['t_s_keplerian'] = 1 / df['Omega_k']

df['avg_ecc'] = df.apply(avg_ecc, axis=1)

df['v_phi'] = df.apply(avg_v_phi, axis=1)

df['avg_req'] = df.apply(avg_req, axis=1)

df['r_fb'] = df['r'] * (df['Omega_r'] / df['Omega_k'])**2

df['avg_teq'] = df.apply(avg_teq, axis=1)

output_path = f'Data/data_{model}.csv'
df.to_csv(output_path, index=False, float_format='%.8e')
#print(f"Archivo guardado en: {output_path}")


# ## Control plots of $t_{\rm eq}(r,\theta)$ and $r_{\rm eq}(r,\theta)$

# In[593]:


df = pd.read_csv(f'Data/data_{model}.csv')



df = pd.read_csv(f'Data/data_{model}.csv')
df.sort_values(by='r', inplace=True)


df['H_t'] = 1 / np.abs(np.gradient(np.log(df['avg_teq']), df['r'], edge_order=2))

df['Mfb_dot'] = 2 *  np.pi * df['r']**2 * df['rho_r'] * avg_norm_factor * (df['H_t'] / df['avg_teq'])

df['M_BH'] =  cumtrapz(df['Mfb_dot'], x=df['avg_teq'], initial=0)

df['integrand'] =  2 *  np.pi * df['r']**4 * df['rho_r'] * df['Omega_r'] * avg_norm_factor
df['J_r'] = cumtrapz(df['integrand'], df['r'], initial=0)
df.drop(columns=['integrand'], inplace=True)

df['a_star'] = (c * df['J_r']) / (G * df['M_r']**2)

# Guardar el DataFrame en un archivo CSV usando el nombre del modelo
output_path = f'Data/data_{model}.csv'
df.to_csv(output_path, index=False, float_format='%.8e')
#print(f"Archivo guardado en: {output_path}")




df = pd.read_csv(f'Data/data_{model}.csv')
df.sort_values(by='r', inplace=True)

def R_isco(M_BH, a_star):
    z1 = 1 + (1 - a_star**2)**(1/3) * ((1 + a_star)**(1/3) + (1 - a_star)**(1/3))
    z2 = np.sqrt(3 * a_star**2 + z1**2)
    return (G * M_BH / c**2) * (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2)))

df['R_isco'] = df.apply(lambda row: R_isco(row['M_BH'], row['a_star']), axis=1)
df['Jfb_dot'] = df['Mfb_dot'] * df['r_fb'] * df['v_phi'].abs()


# Guardar el DataFrame en un archivo CSV usando el nombre del modelo
output_path = f'Data/data_{model}.csv'
df.to_csv(output_path, index=False, float_format='%.8e')
#print(f"Archivo guardado en: {output_path}")


# # Control plots of the previously defined functions

# In[596]:


df = pd.read_csv(f'Data/data_{model}.csv')
df = df.iloc[20:].reset_index(drop=True)

# Ensure t_eq is in the correct numeric format
df['avg_teq'] = pd.to_numeric(df['avg_teq'], errors='coerce')




Jfb_dot_interp = interp1d(df['avg_teq'], df['Jfb_dot'], bounds_error=False, fill_value="extrapolate")



if not profile_files:
    raise FileNotFoundError(f"No se encontraron archivos 'profile*.data' en {logs_dir}")

# Cargar el último perfil
latest_profile = profile_files[-1]
profile_data = MesaData(latest_profile)

# Extraer datos relevantes del perfil MESA
mass_enclosed = profile_data.mass  # Masa en masas solares
log_j_rot = profile_data.log_j_rot  # log del momento angular de rotación

# Radio de Schwarzschild (R_s = 2GM/c^2)
df['R_s'] = 2 * G * df['M_r'] / c**2 

df['j_schwarzschild'] = np.sqrt( 6 * G * df['M_r'] * df['R_s'])
df['j_kerr_max'] = np.sqrt(G * df['M_r'] * (G * df['M_r'] / c**2))
df['j_isco'] = np.sqrt(G * df['M_r'] * df['R_isco'])  
# Ruta base para los datos
data_path = f'Data/data_{model}.csv'
output_path = '../plots'  # Directorio de salida para los gráficos
os.makedirs(output_path, exist_ok=True)  # Crear el directorio si no existe

# Leer el archivo CSV
df = pd.read_csv(data_path)

# No eliminar filas al inicio
remove = 0.0  # Porcentaje de eliminación
remove_rows = int(len(df) * remove)
df = df.iloc[remove_rows:].reset_index(drop=True)

# Crear funciones de interpolación para r_fb y R_isco solo sobre rangos válidos (donde los valores no son NaN)
R_isco_valid_range = df['avg_teq'][df['R_isco'].notna()]
rfb_r_valid_range = df['avg_teq'][df['r_fb'].notna()]

R_isco_interp = interp1d(R_isco_valid_range, df['R_isco'][df['R_isco'].notna()], kind='cubic')
rfb_r_interp = interp1d(rfb_r_valid_range, df['r_fb'][df['r_fb'].notna()], kind='cubic')

# Definir el rango mínimo y máximo para buscar la intersección
mint = max(df['avg_teq'].min(), R_isco_valid_range.min(), rfb_r_valid_range.min())
maxt = min(df['avg_teq'].max(), R_isco_valid_range.max(), rfb_r_valid_range.max())

# Función para calcular la diferencia entre r_fb y R_isco en un t_eq dado
def intersection_eq(t_eq):
    return rfb_r_interp(t_eq) - R_isco_interp(t_eq)

# Buscar múltiples intersecciones
intersections = []
t_eq_range = np.linspace(mint, maxt, 1000)

for i in range(1, len(t_eq_range)):
    if np.sign(intersection_eq(t_eq_range[i - 1])) != np.sign(intersection_eq(t_eq_range[i])):
        # Encontrar el punto de intersección usando root_scalar
        result = root_scalar(intersection_eq, bracket=[t_eq_range[i - 1], t_eq_range[i]])
        if result.converged:
            intersections.append(result.root)
            # Salir del bucle si se encuentran dos intersecciones
            if len(intersections) == 2:
                break

# Manejar los casos en que se encuentran una o dos intersecciones
if len(intersections) >= 2:
    # Usar la segunda intersección
    intersection = intersections[1]
    print(f"Second intersection found at t_eq: {intersection:.2f} s")
elif len(intersections) == 1:
    # Si solo hay una intersección, usar la primera
    intersection = intersections[0]
    print(f"Only one intersection found at t_eq: {intersection:.2f} s")
else:
    print("No intersections found.")
    intersection = None

if intersection is not None:
    # Encontrar el índice más cercano en el DataFrame al punto de intersección
    closest_index = (df['avg_teq'] - intersection).abs().idxmin()
    closest_row = df.iloc[closest_index]

    # Extraer valores críticos
    r_crit = closest_row['r']
    a_star_crit = closest_row['a_star']
    M_BH_crit = closest_row['M_BH'] / M_sun  # Convertir M_BH de gramos a masas solares

# Calcular columnas necesarias si no existen
c = 2.99792458e10  # Velocidad de la luz en cm/s
G = 6.67430e-8  # Constante gravitacional en cgs
if 'j_schwarzschild' not in df.columns:
    df['R_s'] = 2 * G * df['M_r'] / c**2  # Radio de Schwarzschild
    df['j_schwarzschild'] = np.sqrt(6 * G * df['M_r'] * df['R_s'])
if 'j_kerr_max' not in df.columns:
    df['j_kerr_max'] = np.sqrt(G * df['M_r'] * (G * df['M_r'] / c**2))
if 'j_isco' not in df.columns:
    df['j_isco'] = np.sqrt(G * df['M_r'] * df['R_isco'])

##############################################################################################################3333
# Gráfico combinado de LSO + r_fb/Risco: Combined_LSO_plot     
##############################################################################################################3333

# Asegurarse de que las constantes están definidas
G = 6.67430e-8  # Constante gravitacional en cgs
c = 2.99792458e10  # Velocidad de la luz en cm/s
M_sun = 1.989e33  # Masa solar en g

# Variables dinámicas
df = pd.read_csv(f'Data/data_{model}.csv')
df.sort_values(by='r', inplace=True)

# Calcular columnas necesarias si no existen
if 'j_schwarzschild' not in df.columns:
    df['R_s'] = 2 * G * df['M_r'] / c**2  # Radio de Schwarzschild
    df['j_schwarzschild'] = np.sqrt(6 * G * df['M_r'] * df['R_s'])
if 'j_kerr_max' not in df.columns:
    df['j_kerr_max'] = np.sqrt(G * df['M_r'] * (G * df['M_r'] / c**2))
if 'j_isco' not in df.columns:
    df['j_isco'] = np.sqrt(G * df['M_r'] * df['R_isco'])

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 8))

# Graficar los momentos angulares
ax.plot(mass_enclosed, log_j_rot, 
             label=f'Specific angular momentum at LSO', 
             color='purple', linestyle='-', linewidth=1.5, alpha=1.0)
ax.plot(df['M_r'] / M_sun, np.log10(df['j_schwarzschild']), 
        label=r'$j$ $(a_* = 0.00)$ LSO (Schwarzschild)', 
        linestyle='-.', color='green', linewidth=1.0, alpha=0.8)
ax.plot(df['M_r'] / M_sun, np.log10(df['j_isco']), 
        label=fr'$j$ $(a_* = {a_star_crit:.2f})$ LSO using critical $a_*$', 
        linestyle='-', color='red', linewidth=1.5, alpha=1.0)
ax.plot(df['M_r'] / M_sun, np.log10(df['j_kerr_max']), 
        label=r'$j$ $(a_* = 1.00)$ LSO (Kerr)', 
        linestyle='--', color='darkorange', linewidth=1.0, alpha=0.8)

# Etiquetas y personalización del gráfico
ax.set_xlabel(r'$M_r$ ($M_\odot$)', fontsize=15)
ax.set_ylabel(r'$\log \, j$ ($\rm cm^2\ s^{-1}$)', fontsize=15)
ax.set_xlim([0, df['M_r'].max() / M_sun])  # Ajustar límites de x dinámicamente
ax.set_ylim([14, np.ceil(np.log10(df[['j_schwarzschild', 'j_kerr_max', 'j_isco']].max().max()))])
ax.grid(True, which='both', linestyle='--', linewidth=0.7)
ax.legend(loc='lower right', fontsize=14, frameon=True)
ax.set_title(fr'{int(mass)} $M_\odot$; $\Omega/\Omega_{{\rm crit}}={rotation}$; $\eta_{{\rm D}}={eta_d}$', fontsize=22)

# Agregar texto con valores críticos
ax.text(0.08, 0.15,  # Ajusta las coordenadas para cambiar la ubicación
        fr'$a_* = {a_star_crit:.2f}$' + '\n' +
        fr'$M_{{\rm BH}} = {M_BH_crit:.2f} \ M_\odot$', 
        transform=ax.transAxes, fontsize=18, 
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

# Guardar el gráfico
output_plot_path = f'plots/LSO_{model}.pdf'
os.makedirs('plots', exist_ok=True)
plt.savefig(output_plot_path, format="pdf")
#plt.show()

print(f"Gráfico guardado en: {output_plot_path}")














'''
##############################################################################################################3333
# Gráfico combinado de LSO + r_fb/Risco: Combined_LSO_plot     
##############################################################################################################3333

# Graficar r_fb y R_isco vs. t_eq con el punto de intersección marcado
fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # Ajuste de tamaño

# Primer gráfico: r_fb y R_isco
axes[0].loglog(df['avg_teq'], df['r_fb'], label=r"$r_{\rm fb}$", color="magenta", linewidth=1.4)
axes[0].loglog(R_isco_valid_range, R_isco_interp(R_isco_valid_range), label=r"$R_{\rm isco}$", color="darkcyan", linewidth=1.4)
if intersection is not None:
    axes[0].scatter([intersection], [rfb_r_interp(intersection)], color='red', zorder=5, s=100)
    axes[0].text(0.08, 0.35,  # Ajusta las coordenadas para cambiar la ubicación
                 fr'$r_{{\rm crit}} = {r_crit:.2e}$ cm' + '\n' +
                 fr'$a_* = {a_star_crit:.2f}$' + '\n' +
                 fr'$M_{{\rm BH}} = {M_BH_crit:.2f} \ M_\odot$' + '\n' +
                 fr'$t_{{\rm crit}} = {intersection:.2f}$ s',
                 transform=axes[0].transAxes, fontsize=14, 
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
axes[0].set_xlabel(r'$t_{\rm eq}$ (s)', fontsize=15)
axes[0].set_ylabel('r (cm)', fontsize=15)
axes[0].set_ylim(1e3, max(df['r_fb'].max(), df['R_isco'].max()) * 1.2)
axes[0].legend(loc='lower right', fontsize=14, frameon=True)
axes[0].grid(True, which='both', linestyle='--', linewidth=0.7)

# Segundo gráfico: Momentos angulares
axes[1].plot(mass_enclosed, log_j_rot, 
             label=f'Specific angular momentum at LSO', 
             color='purple', linestyle='-', linewidth=1.5, alpha=1.0)
axes[1].plot(df['M_r'] / M_sun, np.log10(df['j_schwarzschild']), 
             label=r'$j$ $(a_* = 0.00)$ LSO (Schwarzschild)', 
             linestyle='-.', color='green', linewidth=1.0, alpha=0.8)
axes[1].plot(df['M_r'] / M_sun, np.log10(df['j_isco']), 
             label=fr'$j$ $(a_* = {a_star_crit:.2f})$ LSO using critical $a_*$', 
             linestyle='-', color='red', linewidth=1.5, alpha=1.0)
axes[1].plot(df['M_r'] / M_sun, np.log10(df['j_kerr_max']), 
             label=r'$j$ $(a_* = 1.00)$ LSO (Kerr)', 
             linestyle='--', color='darkorange', linewidth=1.0, alpha=0.8)
axes[1].set_xlabel(r'$M_r$ ($M_\odot$)', fontsize=15)
axes[1].set_ylabel(r'$\log \, j$ ($\rm cm^2\ s^{-1}$)', fontsize=15)
axes[1].set_xlim([0, df['M_r'].max() / M_sun])  # Ajustar límites de x dinámicamente
axes[1].set_ylim([14, np.ceil(np.log10(df[['j_schwarzschild', 'j_kerr_max', 'j_isco']].max().max()))])
axes[1].grid(True, which='both', linestyle='--', linewidth=0.7)
axes[1].legend(loc='lower right', fontsize=14, frameon=True)
axes[1].set_title(fr'{int(mass)} $M_\odot$ and $\Omega/\Omega_{{\rm crit}}={rotation}$, $\eta_{{\rm D}}={eta_d}$', fontsize=22)

# Configurar ticks personalizados para el eje x en intervalos de 0.5 M_sun
axes[1].set_xticks(np.arange(0, df['M_r'].max() / M_sun + 1, 1))
axes[1].grid(True, which='both', linestyle='--', linewidth=0.7)
for i, label in enumerate(axes[1].get_xticklabels()):
    if i % 5 != 0:  # Saltar etiquetas no múltiplos de 5
        label.set_visible(False)
# Asegurar que el grid sigue activo


# Ajustar proporciones
fig.subplots_adjust(hspace=0.5)  # Mayor separación entre subplots

axes[1].set_position([0.1, 0.46, 0.86, 0.45])  # Ajustar el subplot superior
axes[0].set_position([0.1, 0.1, 0.86, 0.25])  # Ajustar el subplot inferior

# Ajustar márgenes generales y espacio entre subplots


# Guardar la figura
output_plot_path = f'plots/Combined_LSO_{model}.pdf'
#plt.show()

plt.savefig(output_plot_path, format="pdf")

print(f"Gráficos guardados en: {output_plot_path}")

#
'''

# Registrar el fin de la simulación
end_time = time.time()
execution_time = end_time - start_time

# Imprimir el tiempo total de ejecución
print(f"Tiempo total de ejecución: {execution_time:.2f} segundos")