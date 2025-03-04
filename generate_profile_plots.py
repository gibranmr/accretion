#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import matplotlib.pyplot as plt
import mesa_reader as mr
from scipy.interpolate import interp1d
import matplotlib.lines as mlines
import glob
import time

# Registrar el inicio de la simulación
start_time = time.time()
#Parámetros iniciales
rotation = 0.9
eta_d = 1.0


# Directorio base de las simulaciones (dinámico a partir de eta_d)
base_path = f'/fs/phaethon/other0/gibran/sim_results/Si28_5d-2_W9_D{eta_d:.1f}/'


# Generar nombres de simulaciones para masas de 20 a 100 en pasos de 5
masses = range(20, 105, 5)
simulations = [f'{mass}M_W{rotation}_D{eta_d:.1f}' for mass in masses]

colors = [
    'green', 'crimson', 'blue', 'dimgrey', 'springgreen', 'purple', 'yellow', 'cyan', 'magenta', 'brown', 'lime', 
    'teal', 'darkorange', 'red', 'black', 'olive', 'royalblue'
]
# Crear directorio de salida para gráficos
output_path = os.path.join(base_path, '../plots')
os.makedirs(output_path, exist_ok=True)


# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

R_sun_cm = 6.957e10  # 1 solar radius in cm


# In[30]:


def gamma_interpolated():
    gamma_data = np.array([
        [0.7006687633820481, 8.590361734212374],
        [0.7406210974303346, 8.59912053097071],
        [0.7668218336623823, 8.60238956213661],
        [0.7963237252738782, 8.604791013394294],
        [0.8239676504762193, 8.608559864412078],
        [0.854391559337749, 8.612020838255967],
        [0.8831090079107848, 8.616103536354132],
        [0.928500053857564, 8.620516306308115],
        [0.9556233115667899, 8.62241516385847],
        [0.9930241373396642, 8.62806841889161],
        [1.045178733313484, 8.63123924285909],
        [1.0840017428183175, 8.634147279779766],
        [1.115429572575449, 8.636923769157175],
        [1.159084835691005, 8.642586536704426],
        [1.220221327679386, 8.64738948845892],
        [1.2542602092694146, 8.65329913547712],
        [1.343569782107448, 8.672795961282233],
        [1.3950376913438516, 8.682878028680562],
        [1.4484771734129014, 8.691210885054108],
        [1.4979612928603598, 8.697565366473723],
        [1.567833217228917, 8.713198464040786],
        [1.6287601900782025, 8.722696872583056],
        [1.7024622466239647, 8.7355291782133],
        [1.7753293139935402, 8.74818819934191],
        [1.8550701018603537, 8.760772448382957],
        [1.9293184086879398, 8.773370243468937],
        [1.9945912144321303, 8.782421162966724],
        [2.084291515389673, 8.798046079545427],
        [2.157221559890761, 8.80811130432585],
        [2.2524333830464345, 8.824419600830469],
        [2.3462111357563886, 8.839316625537075],
        [2.450940161809495, 8.85241599554133],
        [2.554480384808679, 8.86951042998573],
        [2.7059027032556093, 8.89205650299771],
        [2.8304580274055464, 8.908070829227034],
        [2.946886723652477, 8.925529996259021],
        [3.073999685177946, 8.942637054401812],
        [3.5477249725462996, 9.005395514673028],
        [5.603235400092118, 9.343271151313758],
        [5.7806385229261075, 9.397405946619989],
        [5.893930028557216, 9.430938675806587],
        [4.466222542022773, 9.139701909751727],
        [5.184185251205373, 9.256595136072756]
    ])

    # Interpolación de la línea de \(\Gamma = 4/3\)
    log_rhoc = gamma_data[:, 0]
    log_tc = gamma_data[:, 1]
    gamma_interp = interp1d(log_rhoc, log_tc, kind='linear', bounds_error=False, fill_value='extrapolate')

    return gamma_interp

gamma_interp = gamma_interpolated()


##############################################################################################################3333
# Gráfico de T_CORE vs RHO_CORE
##############################################################################################################3333

# Configuración de la gráfica
plt.figure(figsize=(12, 8))

# Iterar a través de las simulaciones y colores
for sim, color, mass in zip(simulations, colors, masses):
    # Ruta del archivo history.data
    history_path = os.path.join(base_path, sim, 'history.data')
    print(f"Buscando history.data en: {history_path}")

    if os.path.exists(history_path):
        # Cargar los datos
        history = mr.MesaData(history_path)
        
        # Filtrar puntos iniciales y finales si es necesario (opcional)
        initial_points_to_remove = 10
        final_points_to_remove = 1
        max_index = len(history.log_center_T) - final_points_to_remove
        
        # Graficar log_rho_core vs log_T_core
        plt.plot(history.log_center_Rho[initial_points_to_remove:max_index],
                 history.log_center_T[initial_points_to_remove:max_index],
                 label=fr'$M={mass}\ M_{{\odot}}$', 
                 color=color, linewidth=1.5, alpha=0.7)
    else:
        print(f"Advertencia: history.data no encontrado para {sim}")

# Opcional: interpolar y sombrear región donde \(\Gamma < 4/3\)
log_rho_range = np.linspace(0.1, 9, 1000)
# Suponiendo que gamma_interp es una función predefinida
log_t_interp = gamma_interp(log_rho_range)
plt.plot(log_rho_range, log_t_interp, color='red', linestyle='--')
plt.fill_between(log_rho_range, log_t_interp, 10, color='pink', alpha=0.4)

# Configuración de los límites y título dinámico
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'$\log (\rho_\mathrm{core})\ \mathrm{(g\ cm^{-3})}$', fontsize=18)
plt.ylabel(r'$\log (T_\mathrm{core})\ \mathrm{(K)}$', fontsize=18)
# Título dinámico basado en variables
plt.title(
    rf'$\Omega/\Omega_{{\rm crit}} = {rotation}$; $\eta_{{\rm Dutch}}={eta_d}$',
    fontsize=22
)

plt.ylim(7, 10)
plt.xlim(0.1, 8)
plt.text(2.2, 9.43, r"$\Gamma < 4/3$", fontsize=26, color='red')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='lower right', fontsize=12, ncol=3, frameon=True, shadow=True)
plt.tight_layout()
plt.text(2.2, 9.43, r"$\Gamma < 4/3$", fontsize=26, color='red')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# Guardar y mostrar la gráfica con nombre dinámico
output_file = os.path.join(output_path, f'Tc_Rhoc_W{rotation}_D{eta_d}.pdf')
plt.savefig(output_file, format="pdf")
#plt.show()

print(f"Gráfico guardado en: {output_file}")


##############################################################################################################3333
# Gráfico de W_SURF/W_CORE vs AGE  
##############################################################################################################3333

# Configuración de la gráfica
plt.figure(figsize=(12, 8))
# Listas para leyendas
lines = []
labels = []

# Iterar a través de las simulaciones y colores
for sim, color, mass in zip(simulations, colors, masses):
    # Ruta del archivo history.data
    history_path = os.path.join(base_path, sim, 'history.data')
    if os.path.exists(history_path):
        # Cargar los datos
        history = mr.MesaData(history_path)
        
        # Filtrar los últimos puntos si es necesario
        points_to_remove = 1
        max_index = len(history.star_age) - points_to_remove
        age = history.star_age[:max_index]

        # Extraer datos de rotación superficial y del núcleo
        W_Wc_surf = history.surf_avg_omega_div_omega_crit[:max_index]
        W_Wc_core = history.center_omega_div_omega_crit[:max_index]

        # Graficar rotación superficial y del núcleo
        surf_line, = plt.loglog(age, W_Wc_surf, color=color, linestyle='dotted', linewidth=1.0, alpha=0.9)
        core_line, = plt.loglog(age, W_Wc_core, color=color, linestyle='-', linewidth=1.0, alpha=0.9)

        # Guardar las líneas para la leyenda
        lines.append(core_line)
        labels.append(f'$M={mass}\ M_{{\odot}}$')
    else:
        print(f"Advertencia: history.data no encontrado para {sim}")

# Crear leyendas
legend1 = plt.legend(lines, labels, loc='lower left', fontsize=12, frameon=True, shadow=True, ncol=3)
plt.gca().add_artist(legend1)

solid_line = mlines.Line2D([], [], color='black', linestyle='-', label='Core Rotation')
dotted_line = mlines.Line2D([], [], color='black', linestyle='dotted', label='Surface Rotation')
plt.legend(handles=[solid_line, dotted_line], loc='best', fontsize=18, frameon=True, shadow=True)

# Configuración del gráfico
plt.ylim(0.5e-3, 2e0)
plt.xlim(4e5, 1.5e7)
plt.xlabel(r'${\rm Age}$ ${\rm (yr)}$', fontsize=18)
plt.ylabel(r'$\Omega/\Omega_{\mathrm{crit}}$', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(
    rf'$\Omega/\Omega_{{\rm crit}} = {rotation}$; $\eta_{{\rm Dutch}}={eta_d}$',
    fontsize=22
)
plt.grid(True, which='both', linestyle='-.', linewidth=0.5)


# Guardar y mostrar la gráfica con nombre dinámico
output_file = os.path.join(output_path, f'Omega_Age_W{rotation}_D{eta_d}.pdf')
plt.savefig(output_file, format="pdf")
#plt.show()

print(f"Gráfico guardado en: {output_file}")


##############################################################################################################3333
# Gráfico de M_DOT vs AGE  
##############################################################################################################3333


# Configuración general de la figura
plt.figure(figsize=(12, 8))

# Iterar a través de las simulaciones y colores
for sim, color, mass in zip(simulations, colors, masses):
    # Ruta del archivo history.data
    history_path = os.path.join(base_path, sim, 'history.data')
    if os.path.exists(history_path):
        # Cargar los datos
        history = mr.MesaData(history_path)
        
        # Filtrar puntos finales si es necesario
        points_to_remove = 1
        max_index = len(history.star_age) - points_to_remove
        
        # Extraer edad y \(\dot{M}\)
        age = history.star_age[:max_index]
        mdot = history.log_abs_mdot[:max_index]  # log10(|Mdot|)

        # Graficar \(\log |\dot{M}|\) vs \(\log (\text{Edad})\)
        plt.plot(np.log10(age), mdot, label=f'$M={mass}\ M_{{\odot}}$', color=color, linestyle='-', linewidth=1.0, alpha=0.7)
    else:
        print(f"Advertencia: history.data no encontrado para {sim}")

# Configuración del gráfico
plt.xlim(5.6, 7.2)
plt.ylim(-11, -2)
plt.xlabel(r'$\log$ ${\rm Age}$ ${\rm (yr)}$', fontsize=18)
plt.ylabel(r'$\log |\dot{M}|\ \ (M_\odot~\mathrm{yr}^{-1})$', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(
    rf'$\Omega/\Omega_{{\rm crit}} = {rotation}$; $\eta_{{\rm Dutch}}={eta_d}$',
    fontsize=22
)
plt.grid(True, which='both', linestyle='-.', linewidth=0.5)

# Leyenda
plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, ncol=3)


# Guardar y mostrar la gráfica con nombre dinámico
output_file = os.path.join(output_path, f'Mdot_Age_W{rotation}_D{eta_d}.pdf')
plt.savefig(output_file, format="pdf")
#plt.show()

print(f"Gráfico guardado en: {output_file}")

##############################################################################################################3333
# Gráfico de LOG RHO vs RADIUS  
##############################################################################################################3333

plt.figure(figsize=(12, 8))

for sim, color, mass in zip(simulations, colors, masses):
    logs_dir = os.path.join(base_path, sim)
    profile_file = glob.glob(os.path.join(logs_dir, 'profile*.data'))
    if profile_file:  # Verificar que exista al menos un archivo
        profile_file = profile_file[0]  # Obtener el único archivo
        profile_data = mr.MesaData(profile_file)

        radius = profile_data.radius * R_sun_cm
        log_rho = profile_data.logRho

        plt.plot(radius, log_rho, label=f'$M={mass}\ M_\\odot$', color=color, linestyle='-', linewidth=1.2, alpha=0.8)
    else:
        print(f"Advertencia: No se encontró archivo profile*.data en {logs_dir}")

plt.xscale('log')
plt.xlabel(r'$r$ (cm)', fontsize=18)
plt.ylabel(r'$\log\ \rho$ ($\rm g\ cm^{-3}$)', fontsize=18)
plt.ylim(0,8)
plt.xlim(4e6,4e10)
plt.title(
    rf'$\Omega/\Omega_{{\rm crit}} = {rotation}$; $\eta_{{\rm Dutch}}={eta_d}$',
    fontsize=22
)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='lower left', fontsize=14, frameon=True, shadow=True, ncol=3)
plt.tight_layout()

# Guardar y mostrar la gráfica con nombre dinámico
output_file = os.path.join(output_path, f'Rho_R_W{rotation}_D{eta_d}.pdf')
plt.savefig(output_file, format="pdf")
#plt.show()

print(f"Gráfico guardado en: {output_file}")

##############################################################################################################3333
# Gráfico de LOG RHO vs M_R
##############################################################################################################3333


plt.figure(figsize=(12, 8))

for sim, color, mass in zip(simulations, colors, masses):
    logs_dir = os.path.join(base_path, sim)
    profile_file = glob.glob(os.path.join(logs_dir, 'profile*.data'))
    if profile_file:  # Verificar que exista al menos un archivo
        profile_file = profile_file[0]  # Obtener el único archivo
        profile_data = mr.MesaData(profile_file)

        mass_enclosed = profile_data.mass
        log_rho = profile_data.logRho

        plt.plot(mass_enclosed, log_rho, label=f'$M={mass}\ M_\\odot$', color=color, linestyle='-', linewidth=1.2, alpha=0.8)
    else:
        print(f"Advertencia: No se encontró archivo profile*.data en {logs_dir}")

plt.xlabel(r'$M_r$ ($M_\odot$)', fontsize=18)
plt.ylabel(r'$\log\ \rho$ ($\rm g\ cm^{-3}$)', fontsize=18)
plt.title(
    rf'$\Omega/\Omega_{{\rm crit}} = {rotation}$; $\eta_{{\rm Dutch}}={eta_d}$',
    fontsize=22
)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(2,8)
plt.legend(loc='best', fontsize=12, frameon=True, shadow=True, ncol=3)
plt.tight_layout()

# Guardar y mostrar la gráfica con nombre dinámico
output_file = os.path.join(output_path, f'Rho_Mr_W{rotation}_D{eta_d}.pdf')
plt.savefig(output_file, format="pdf")
#plt.show()

print(f"Gráfico guardado en: {output_file}")
#plt.show()

##############################################################################################################3333
# Gráfico de LOG OMEGA vs R 
##############################################################################################################3333

plt.figure(figsize=(12, 8))

for sim, color, mass in zip(simulations, colors, masses):
    logs_dir = os.path.join(base_path, sim)
    profile_file = glob.glob(os.path.join(logs_dir, 'profile*.data'))
    if profile_file:  # Verificar que exista al menos un archivo
        profile_file = profile_file[0]  # Obtener el único archivo
        profile_data = mr.MesaData(profile_file)

        radius = profile_data.radius * R_sun_cm
        log_omega = np.log10(profile_data.omega)

        plt.plot(radius, log_omega, label=f'$M={mass}\ M_\\odot$', color=color, linestyle='-', linewidth=1.2, alpha=0.8)
    else:
        print(f"Advertencia: No se encontró archivo profile*.data en {logs_dir}")

plt.xscale('log')
#plt.xlim(3e7,1e11)
#plt.ylim(-4.5, -1)
plt.xlim(3e7,2e11)   ##Solo para el caso eta_d = 0.1
plt.ylim(-3.5, -1)   ##Solo para el caso eta_d = 0.1

plt.xlabel(r'$r$ (cm)', fontsize=18)
plt.ylabel(r'$\log\ (\Omega)$ (rad/s)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(
    rf'$\Omega/\Omega_{{\rm crit}} = {rotation}$; $\eta_{{\rm Dutch}}={eta_d}$',
    fontsize=22
)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='lower left', fontsize=14, frameon=True, shadow=True, ncol=3)
plt.tight_layout()

# Guardar y mostrar la gráfica con nombre dinámico
output_file = os.path.join(output_path, f'Omega_R_W{rotation}_D{eta_d}.pdf')
plt.savefig(output_file, format="pdf")
#plt.show()

print(f"Gráfico guardado en: {output_file}")


##############################################################################################################3333
# Gráfico de LOG J vs M_r  
##############################################################################################################3333


# Plot: log(j) vs Enclosed Mass (M_r)
plt.figure(figsize=(12, 8))

for sim, color, mass in zip(simulations, colors, masses):
    logs_dir = os.path.join(base_path, sim)
    profile_file = glob.glob(os.path.join(logs_dir, 'profile*.data'))
    if profile_file:  # Verificar que exista al menos un archivo
        profile_file = profile_file[0]  # Obtener el único archivo
        profile_data = mr.MesaData(profile_file)

        mass_enclosed = profile_data.mass  # M_r en masas solares
        log_j_rot = profile_data.log_j_rot  # Log del momento angular específico

        plt.plot(mass_enclosed, log_j_rot, label=f'$M={mass}\ M_\\odot$', color=color, linestyle='-', linewidth=1.2, alpha=0.8)
    else:
        print(f"Advertencia: No se encontró archivo profile*.data en {logs_dir}")

# Configuración del gráfico
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(14,17.5)
plt.xlabel(r'$M_r$ ($M_\odot$)', fontsize=18)
plt.ylabel(r'$\log \ j$ ($\rm cm^2\ s^{-1}$)', fontsize=18)
plt.title(
    rf'$\Omega/\Omega_{{\rm crit}} = {rotation}$; $\eta_{{\rm Dutch}}={eta_d}$',
    fontsize=22
)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=14, frameon=True, shadow=True, ncol=3)
plt.tight_layout()


# Guardar y mostrar la gráfica con nombre dinámico
output_file = os.path.join(output_path, f'J_Mr_W{rotation}_D{eta_d}.pdf')
plt.savefig(output_file, format="pdf")
#plt.show()

print(f"Gráfico guardado en: {output_file}")


# Registrar el fin de la simulación
end_time = time.time()
execution_time = end_time - start_time

# Imprimir el tiempo total de ejecución
print(f"Tiempo total de ejecución: {execution_time:.2f} segundos")

#
##############################################################################################################3333
# Gráfico de la evolución temporal de la masa total de la estrella
##############################################################################################################3333

plt.figure(figsize=(12, 8))

# Iterar sobre cada simulación, usando los mismos colores y etiquetas definidos
for sim, color, mass in zip(simulations, colors, masses):
    # Ruta del archivo history.data para la simulación actual
    history_path = os.path.join(base_path, sim, 'history.data')
    
    if os.path.exists(history_path):
        # Cargar datos del historial con mesa_reader
        history = mr.MesaData(history_path)
        
        # Extraer la edad y la masa total de la estrella
        age = history.star_age      # Edad estelar en años
        total_mass = history.star_mass  # Masa total en M_sol
        
        # Graficar la evolución temporal de la masa total
        plt.plot(np.log10(age), total_mass, label=f'$M={mass}\ M_\\odot$', color=color,
                 linestyle='-', linewidth=1.2, alpha=0.8)
    else:
        print(f"Advertencia: history.data no encontrado para {sim}")

# Configuración del gráfico
plt.xlabel(r'$\log \rm Age\ (yr)$', fontsize=18)
plt.xlim(6.4, 7.1)
plt.ylabel(r'$M_{\rm star}\ (M_\odot)$', fontsize=18)
plt.title(rf'$\Omega/\Omega_{{\rm crit}} = {rotation}$; $\eta_{{\rm Dutch}} = {eta_d}$', fontsize=22)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, which='both', linestyle='-.', linewidth=0.5)
plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, ncol=3)
# Guardar y mostrar la gráfica
output_file = os.path.join(output_path, f'Mass_Age_W{rotation}_D{eta_d}.pdf')
plt.savefig(output_file, format="pdf")
plt.show()

print(f"Gráfico guardado en: {output_file}")