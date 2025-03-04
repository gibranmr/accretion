import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps
from scipy.optimize import newton
from scipy.optimize import root_scalar
from scipy.integrate import quad
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
mpl.rc('text', usetex=True)
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.unicode'] = True  # Uncomment this if needed for older matplotlib versions
mpl.rcParams.update({'font.size': 18})

# Constants
G = 6.67430e-8  # Gravitational constant in cgs units
M_sun = 1.989e33  # Solar mass in g
#M = 14 * M_sun
# N_n_value: 0.424216926041877
# N_n_value = (4 * np.pi)**(1/n) / (n + 1) * (-xi_1**2 * theta_prime_xi_1)**((1 - n) / n) * xi_1**((n - 3) / n)
# K = N_n_value * G * R * M**(1/3)
#K = 4.44540243e+14 # # Polytropic constant K (K = N_n_value * G * R * M**(1/3) in cgs units
gamma = 5/3 # Adiabatic index

"""
R_star = 5.18e10
# I have used WebPlotDigitizer website in order to extract the data for the built-in figures in the PDF
# Loading the CSV data for Omega

data_Omega = pd.read_csv('./Data/log_Omega_vs_r.csv', header=None, sep=";", decimal=",")
radius_values_Omega = data_Omega[0].values
log_Omega = data_Omega[1].values
# Building an interpolation function for Omega as a function of radius
Omega_function = interp1d(radius_values_Omega, 10**log_Omega, kind='cubic', fill_value="extrapolate", bounds_error=False)
# Importing a .CSV file that fits $\log(\rho)$ vs $M(r)$

# Loading the CSV data
data_rhoM = pd.read_csv('./Data/log_rho_vs_M_r.csv',header=None,sep=";",decimal=",")
#This data were obtained from Figure 2a of Kumar+2018
M_r_values = data_rhoM[0].values*M_sun
log_density_values = data_rhoM[1].values
density_values = 10**log_density_values   # Convert log density to actual density
# Building an interpolating function for density as a function of M(r)
density_function_M = interp1d( M_r_values, log_density_values, kind='linear', fill_value="extrapolate")
# Importing a .CSV file that fits $\log(\rho)$ vs $r$

# Loading the CSV data
data_rhoR = pd.read_csv('./Data/log_rho_vs_r.csv',header=None,sep=";",decimal=",")
radius_values = data_rhoR[0].values 
log_density_values_r = data_rhoR[1].values
density_values_r = 10**log_density_values_r   # Convert log density to actual density
# Building an interpolating function for density as a function of M(r)
density_function_r = interp1d( radius_values, log_density_values_r, kind='cubic', fill_value="extrapolate")
  
def load_and_interpolate_data(file_path, sep, decimal, column1, column2):
    data = pd.read_csv(file_path, header=None, sep=sep, decimal=decimal)
    return interp1d(data[column1].values, 10**data[column2].values, kind='cubic', fill_value="extrapolate", bounds_error=False)

Omega_function = load_and_interpolate_data('./Data/log_Omega_vs_r.csv', ";", ",", 0, 1)
density_function_r = load_and_interpolate_data('./Data/log_rho_vs_r.csv', ";", ",", 0, 1)
"""

# Using the interpolated Omega function to get the Omega for a given radius
def Omega_interpolated(r):
    log_Omega_val = Omega_function(r)
    return 10**log_Omega_val


def M_r(r):
    result, _ = quad(lambda r_prime: 4 * np.pi * r_prime**2 * rho_interpolated(r_prime), 1e5, r)
    return result


# Keplerian angular velocity
def Omega_k(r):
    return np.sqrt(G * M_r(r) / r**3)

# Eccentricity function
def eccentricity(r, theta):
    Omega = Omega_interpolated(r)
    Inv_Omega_k = 1/Omega_k(r)
    return 1 - (Omega * Inv_Omega_k * np.sin(theta))**2

# Function for r_eq
def r_eq(r, theta):
    e = eccentricity(r, theta)
    return r * (1 - e)

def avg_req(r):
    integrand = lambda theta: r_eq(r, theta) * np.sin(theta)
    integral, _ = quad(integrand, 0, np.pi/2)
    avg_norm_factor = (np.sqrt(3)/2)
    return integral/avg_norm_factor
                

def t_s_keplerian(r):
    return 1 / Omega_k(r)
# Function for the polytropic sound speed c_s(r)
def c_s(r):
    rho_val = rho_interpolated(r)
    return np.sqrt(gamma * K * rho_val**(2/3))
# Function for t_s using polytropic sound speed
def t_s_polytropic(r):
    R_star = 5.18e10
    return R_star / c_s(r)

# Define t_eq function
def t_eq(r, theta):
    e = eccentricity(r, theta)
    Inv_Omega_k = 1 / Omega_k(r)
    parenthesis = (np.arccos(-e) + e * np.sqrt(1 - e**2)) * (1 + e)**(-1.5)
    t_s_val = t_s_polytropic(r)
    t_eq_val = Inv_Omega_k * parenthesis + t_s_val
    return t_eq_val


def avg_teq(r): # It averages ONLY for the northern hemisphere since it's axisymmetric
    integrand = lambda theta: t_eq(r, theta) * np.sin(theta)
    integral, _ = quad(integrand, np.pi/6, np.pi/2)
    avg_norm_factor = (np.sqrt(3)/2)    
    return integral/avg_norm_factor


def t_eq_polytropic(r, theta):
    e = eccentricity(r, theta)
    Omega_k_val = Omega_k(r)
    parenthesis = (np.arccos(-e) + (e * np.sqrt(1 - e**2))) * ((1 + e)**(-1.5))
    return (1 / Omega_k_val) * parenthesis + t_s_polytropic(r)
def t_eq_keplerian(r, theta):
    e = eccentricity(r, theta)
    Omega_k_val = Omega_k(r)
    parenthesis = (np.arccos(-e) + (e * np.sqrt(1 - e**2))) * ((1 + e)**(-1.5))
    return (1 / Omega_k_val) * parenthesis + t_s_keplerian(r)
def t_eq_expanded(r, theta):
    Omega_val = Omega_interpolated(r)
    Omega_k_val = Omega_k(r)
    t_s_val = t_s_keplerian(r)
    parenthesis = ((Omega_val * np.sin(theta)) / Omega_k_val)
    return t_s_val + (np.pi / (2**(3/2) * Omega_k_val)) * (1 + (3/4) * parenthesis**2)
def particle_trajectory(t, r, theta, phi, e, M_r):
    Omega_k_val = Omega_k(r).value
    eta = eta_from_t(t, e, r, M_r)

    a = r / (1 + e)

    x = a * (e + np.cos(eta)) * np.sin(theta) * np.cos(phi) + a * (1 - e**2)**0.5 * np.sin(eta) * np.sin(phi)
    y = a * (e + np.cos(eta)) * np.sin(theta) * np.sin(phi) - a * (1 - e**2)**0.5 * np.sin(eta) * np.cos(phi)
    z = a * (e + np.cos(eta)) * np.cos(theta)

    return x, y, z
def eta_from_t(t, e, r):
    Omega_k_val = Omega_k(r).value
    # The equation relating eta to t
    def f(eta):
        return Omega_k_val * t - (eta + e * np.sin(eta)) * (1 + e)**(-1.5)
    return newton(f, 0)
def v_x(r, theta, phi, e):
    return -r * Omega_k(r) * (np.sin(theta) * np.cos(phi) + e * np.sin(phi)) / np.sqrt(1 - e)
def v_y(r, theta, phi, e):
    return -r * Omega_k(r) * (np.sin(theta) * np.sin(phi) - e * np.cos(phi)) / np.sqrt(1 - e)
def v_z(r, theta, e):
    return -np.sign(r) * r * Omega_k(r) * np.cos(theta) / np.sqrt(1 - e)

def avg_v_phi(r):
    integrand = lambda theta: (r * Omega_k(r) * np.sin(theta)) / np.sqrt(1 - eccentricity(r, theta))
    integral, _ = quad(integrand, np.pi/6, np.pi/2)
    avg_norm_factor = (np.sqrt(3)/2)
    return integral / avg_norm_factor

def rho_interpolated(r):
    log_density = density_function_r(r)
    return 10**log_density

def dln_Omega_k_dr(r):
    return -(3/(2*r)) #
def M_dot_fb(r):
    term1 = 4 * np.pi * r**2 * rho_interpolated(r)
    term2 =  (t_eq(r) * np.abs(dln_Omega_k_dr(r)))
    return (term1 / term2)
# Function to find r for a given t_eq using Newton's method
def r_teq_newton(target_t_eq):
    def to_minimize(r):
        return avg_t_eq(r) - target_t_eq

    try:
        return newton(to_minimize, R_star / 2)  # Initial guess at half the star's radius
    except RuntimeError:
        return None

"""   
# It requires to read the "avg_teq_req.csv" dataset which contains three columns: (r,avg_teq,avg_req)
# At the end of the script it is displayed the block-code that it was used to read the dataset
# Read the data
df = pd.read_csv('avg_teq_req.csv')
# Create interpolation functions
avg_teq_interpolated = interp1d(df['r'], df['avg_teq'], kind='cubic', fill_value="extrapolate")
avg_req_interpolated = interp1d(df['r'], df['avg_req'], kind='cubic', fill_value="extrapolate")
r_teq_interpolated = interp1d(df['avg_teq'], df['r'], kind='cubic', fill_value="extrapolate")
# Define the new functions
def avg_teq_i(r):
    return avg_teq_interpolated(r)

def avg_req_i(r):
    return avg_req_interpolated(r)

def r_teq_i(t_eq):
    return r_teq_interpolated(t_eq)



def rfb_teq(t_eq):
    G = 6.67430e-8 
    c = 2.99792458e10 
    M_sun = 1.989e33  
    M = 5.7 * M_sun
    R_schwarzschild = 2 * G * M / c**2
    r = r_teq_interpolated(t_eq)
    r_fb = r_fb_r(r)
    return r_fb / R_schwarzschild 
"""

# Function for r_fb based on Omegas
def r_fb_r(r):
    Omega_val = Omega_interpolated(r)
    Omega_k_val = Omega_k(r)
    return r * (Omega_val / Omega_k_val)**2


def H_t(r):
    avg_teq = avg_teq_i(r)
    # Calculate the derivative of log(avg_teq) with respect to r
    # Since we have a single value, we use a small delta for r to approximate the derivative
    delta_r = 1e-6 * r  # small change in r
    dlog_avg_teq_dr = (np.log(avg_teq_i(r + delta_r)) - np.log(avg_teq)) / delta_r
    # Calculate the inverse of the absolute value of the derivative
    return 1 / np.abs(dlog_avg_teq_dr)

# Function to calculate the mass fallback rate
def dMfb_dt(t_eq):
    r = r_teq_interpolated(t_eq)  # Get the radius corresponding to t_eq
    rho_val = rho_interpolated(r)  # Get the density at this radius
    H_t_val = H_t(r)  # Calculate H_t at this radius
    return (4 * np.pi * r**2 * rho_val * H_t_val) / t_eq

# Define the integrand for the angular momentum integral
def J_r(r):
    integrand = lambda r_value:  4 * np.pi * r_value**4 * rho_interpolated(r_value) * Omega_interpolated(r_value) 
    integral, _ = quad(integrand, 0, r)
    return integral







"""
# ENCLOSED MASS (R)

# Función para calcular la masa encerrada hasta un radio r
def enclosed_mass(r):
    # Integrando la densidad para obtener la masa
    result, _ = quad(lambda r_prime: 4 * np.pi * r_prime**2 * rho_interpolated(r_prime), 0, r)
    return result

# Crear un array de radios para calcular M(r)
radii = np.linspace(min(radius_values), max(radius_values), 1000)
mass_enclosed = np.array([enclosed_mass(r) for r in radii])

# Crear una función interpolada para M(r)
mass_function_r = interp1d(radii, mass_enclosed, kind='cubic', fill_value="extrapolate")
"""

"""
#SOLUTIONS TO LANE EMBDEN EQUATION

import numpy as np
from scipy.integrate import solve_ivp

# Índice politrópico
n = 1.5

# Función que define las ecuaciones diferenciales de Lane-Emden
def lane_emden_derivs(xi, y):
    theta, phi = y
    if xi == 0:
        # Evitar la división por cero
        return [0, 0]
    else:
        return [phi, -(2/xi)*phi - theta**n]

# Condiciones iniciales
y0 = [1, 0]  # theta(0) = 1, theta'(0) = 0

# Resolver la ecuación de Lane-Emden
xi_span = (0, 10)  # xi va de 0 a 10, que es suficiente para encontrar xi_1 para n=1.5
sol = solve_ivp(lane_emden_derivs, xi_span, y0, rtol=1e-10, atol=1e-10)

# Encontrar xi_1 donde theta es cero (o muy cercano a cero)
xi_1_index = np.where(sol.y[0] < 1e-6)[0][0]
xi_1 = sol.t[xi_1_index]
theta_prime_xi_1 = sol.y[1][xi_1_index]

# Calcular (-xi^2 (dtheta/dxi))_xi1
minus_xi2_phi = -xi_1**2 * theta_prime_xi_1

# Calcular la constante numérica N_n
N_n_value = (4 * np.pi)**(1/n) / (n + 1) * (-xi_1**2 * theta_prime_xi_1)**((1 - n) / n) * xi_1**((n - 3) / n)


# Calculamos K
K = N_n_value * G * R * M**(1/3)

print(f"xi_1: {xi_1}")
print(f"theta'(xi_1): {theta_prime_xi_1}")
print(f"-xi2_phi: {minus_xi2_phi}")
print(f"N_n_value: {N_n_value}")
print(f"K  es: {K:.3e}")
"""

"""
# AVERAGE INTERPOLATED FUNCTION
# RADIUS RANGE = (0.5E7,3.72E10) CM
# TEQ RANGE = (0,829) SECONDS


r_values = np.linspace(0.5e7, 3.72e10, 1000)  # Adjust the range as needed
avg_t_eq_values = [avg_t_eq(r) for r in r_values]
avg_r_eq_values = [avg_r_eq(r) for r in r_values]

# Save the data to a CSV file
data = pd.DataFrame({
    'r': r_values,
    'avg_teq': avg_t_eq_values,
    'avg_req': avg_r_eq_values
})
data.to_csv('avg_teq_req.csv', index=False)

data = pd.read_csv('avg_teq_req.csv')

# Create interpolation functions
avg_teq_interpolated = interp1d(data['r'], data['avg_teq'], kind='cubic', fill_value="extrapolate")
avg_req_interpolated = interp1d(data['r'], data['avg_req'], kind='cubic', fill_value="extrapolate")

# Define a range of radius values for plotting
r_plot_range = np.linspace(min(data['r']), max(data['r']), 1000)

# Evaluate the interpolated function over this range
avg_teq_plot_values = avg_teq_interpolated(r_plot_range)

# Plotting avg_teq as a function of r
plt.figure(figsize=(10, 6))
plt.semilogx(r_plot_range, avg_teq_plot_values, label='Average $t_{eq}(r)$')
plt.xlabel('Radius (r) [cm]')
plt.ylabel('Average $t_{eq}$')
plt.title('Average $t_{eq}$ as a Function of Radius')
plt.grid(True)
plt.legend()
plt.show()

# Inverse interpolation: t_eq as independent variable, r as dependent
r_teq_interpolated = interp1d(data['avg_teq'], data['r'], kind='cubic', fill_value="extrapolate")
# Define a range of avg_teq values for plotting
avg_teq_range = np.linspace(min(data['avg_teq']), max(data['avg_teq']), 1000)

# Define a range of avg_teq values for plotting
avg_teq_range = np.linspace(min(data['avg_teq']), max(data['avg_teq']), 1000)

# Evaluate the new function over this range
r_values_from_teq = [r_teq_interpolated(t_eq) for t_eq in avg_teq_range]

# Plotting r as a function of avg_teq
plt.figure(figsize=(10, 6))
plt.loglog(avg_teq_range, r_values_from_teq, label='r($t_{eq}$)')
plt.xlabel('Average $t_{eq}$')
plt.ylabel('Radius (r) [cm]')
plt.title('Radius as a Function of Average $t_{eq}$')
plt.grid(True)
plt.legend()
plt.show()


































# ASTROPHYSICS_DATA.CSV [t_eq_avg, r_eq_avg, r]
from scipy.integrate import quad

# Constants
R_star = 2.9e10  # Example value for the radius of the star in cm

# Define the range of r values
r_values = np.linspace(1e6, R_star, 500)  # Adjust the range and number of points as needed

# Compute t_eq_avg and r_eq_avg for each r
data = []
for r in r_values:
    t_eq_avg_val = t_eq_avg(r)  
    r_eq_avg_val = r_eq_avg(r)  
    data.append([t_eq_avg_val, r_eq_avg_val, r])

# Save the data to a CSV file
df = pd.DataFrame(data, columns=['t_eq_avg', 'r_eq_avg', 'r'])
df.to_csv('astrophysics_data.csv', index=False)

# Read the data
df = pd.read_csv('astrophysics_data.csv')

# Create interpolation functions from t_eq and r_eq
t_eq_avg_interp = interp1d(df['r'], df['t_eq_avg'], kind='cubic', fill_value="extrapolate")
r_eq_avg_interp = interp1d(df['r'], df['r_eq_avg'], kind='cubic', fill_value="extrapolate")


"""



"""
COMPARISON BETWEEN TEQ_K, TEQ_P, REQ_K, REQ_P, R INTERPOLATED 
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Constants
R_star = 2.9e10  # Example value for the radius of the star in cm

def t_eq_keplerian(r, theta):
    e = eccentricity(r, theta)
    return (1 / Omega_k(r)) * (np.arccos(-e) + e * np.sqrt(1 - e**2)) * (1 + e)**(-1.5) + t_s_keplerian(r)

def t_eq_polytropic(r, theta):
    e = eccentricity(r, theta)
    return (1 / Omega_k(r)) * (np.arccos(-e) + e * np.sqrt(1 - e**2)) * (1 + e)**(-1.5) + t_s_polytropic(r)


def t_eq_avg_polytropic(r):
    integrand = lambda theta: t_eq_polytropic(r, theta)
    avg_t_eq, _ = quad(integrand, 0, np.pi/2)
    return avg_t_eq / (np.pi/2)

def t_eq_avg_keplerian(r):
    integrand = lambda theta: t_eq_keplerian(r, theta)
    avg_t_eq, _ = quad(integrand, 0, np.pi/2)
    return avg_t_eq / (np.pi/2)

def r_eq_avg_polytropic(r):
    integrand = lambda theta: r_eq(r, theta)
    avg_r_eq, _ = quad(integrand, 0, np.pi/2)
    return avg_r_eq / (np.pi/2)

def r_eq_avg_keplerian(r):
    integrand = lambda theta: r_eq(r, theta)
    avg_r_eq, _ = quad(integrand, 0, np.pi/2)
    return avg_r_eq / (np.pi/2)

# Define the range of r values
r_values = np.linspace(1e5, R_star, 1000)  # Adjust the range and number of points as needed

# Compute values for each r
data = []
for r in r_values:
    t_eq_avg_polytropic_val = t_eq_avg_polytropic(r)
    t_eq_avg_keplerian_val = t_eq_avg_keplerian(r)
    r_eq_avg_polytropic_val = r_eq_avg_polytropic(r)
    r_eq_avg_keplerian_val = r_eq_avg_keplerian(r)
    data.append([t_eq_avg_polytropic_val, t_eq_avg_keplerian_val, r_eq_avg_polytropic_val, r_eq_avg_keplerian_val, r])

# Save the data to a CSV file
df = pd.DataFrame(data, columns=['t_eq_avg_polytropic', 't_eq_avg_keplerian', 'r_eq_avg_polytropic', 'r_eq_avg_keplerian', 'r'])
df.to_csv('astrophysics_data.csv', index=False)

# Read the data
df = pd.read_csv('astrophysics_data.csv')

# Create interpolation functions
t_eq_avg_polytropic_interp = interp1d(df['r'], df['t_eq_avg_polytropic'], kind='cubic', fill_value="extrapolate")
t_eq_avg_keplerian_interp = interp1d(df['r'], df['t_eq_avg_keplerian'], kind='cubic', fill_value="extrapolate")
r_eq_avg_polytropic_interp = interp1d(df['r'], df['r_eq_avg_polytropic'], kind='cubic', fill_value="extrapolate")
r_eq_avg_keplerian_interp = interp1d(df['r'], df['r_eq_avg_keplerian'], kind='cubic', fill_value="extrapolate")

# Define the interpolated functions
def t_eq_avg_polytropic_interpolated(r):
    return t_eq_avg_polytropic_interp(r)

def t_eq_avg_keplerian_interpolated(r):
    return t_eq_avg_keplerian_interp(r)

def r_eq_avg_polytropic_interpolated(r):
    return r_eq_avg_polytropic_interp(r)

def r_eq_avg_keplerian_interpolated(r):
    return r_eq_avg_keplerian_interp(r)

#======================================================================================0

import matplotlib.pyplot as plt

# Define the range of r values for plotting
r_values = np.linspace(1e5, R_star, 1000)  # Adjust the range and number of points as needed

# Compute interpolated values for plotting
t_eq_avg_polytropic_values = t_eq_avg_polytropic_interpolated(r_values)
t_eq_avg_keplerian_values = t_eq_avg_keplerian_interpolated(r_values)
r_eq_avg_polytropic_values = r_eq_avg_polytropic_interpolated(r_values)
r_eq_avg_keplerian_values = r_eq_avg_keplerian_interpolated(r_values)

# Plotting t_eq_avg
plt.figure(figsize=(10, 6))
plt.semilogx(r_values, t_eq_avg_polytropic_values, label='Polytropic $t_{eq}$')
plt.plot(r_values, t_eq_avg_keplerian_values, label='Keplerian $t_{eq}$')
plt.xlabel(r" r (cm)")
plt.ylabel("$t_{\mathrm eq}$ (s)")
plt.title(r"$t_{\rm eq}$ ")
plt.legend()
plt.grid(True)
plt.show()

# Plotting r_eq_avg
plt.figure(figsize=(10, 6))
plt.loglog(r_values, r_eq_avg_polytropic_values, label='Polytropic $r_{eq}$')
plt.plot(r_values, r_eq_avg_keplerian_values, label='Keplerian $r_{eq}$')
plt.xlabel(r" r (cm)")
plt.ylabel("$r_{\mathrm eq}$ (cm)")
plt.title(r"$r_{\rm eq}$ ")
plt.legend()
plt.grid(True)
plt.show()

#=======================================================================================
# Assuming df has the necessary columns
r_from_t_eq_polytropic_interp = interp1d(df['t_eq_avg_polytropic'], df['r'], kind='cubic', fill_value="extrapolate")
r_from_t_eq_keplerian_interp = interp1d(df['t_eq_avg_keplerian'], df['r'], kind='cubic', fill_value="extrapolate")

def r_teq_polytropic_interpolated(t_eq):
    return r_from_t_eq_polytropic_interp(t_eq)

def r_teq_keplerian_interpolated(t_eq):
    return r_from_t_eq_keplerian_interp(t_eq)

# Define a range of t_eq values for plotting
t_eq_values = np.linspace(min(df['t_eq_avg_polytropic'].min(), df['t_eq_avg_keplerian'].min()), 
                          max(df['t_eq_avg_polytropic'].max(), df['t_eq_avg_keplerian'].max()), 
                          1000)

# Compute r values from t_eq for both cases
r_values_polytropic = r_teq_polytropic_interpolated(t_eq_values)
r_values_keplerian = r_teq_keplerian_interpolated(t_eq_values)

# Plotting R(TEQ)
plt.figure(figsize=(10, 6))
plt.semilogy(t_eq_values, r_values_polytropic, label='Polytropic $r(t_{eq})$')
plt.plot(t_eq_values, r_values_keplerian, label='Keplerian $r(t_{eq})$')
plt.xlabel('$t_{eq}$ (s)')
plt.ylabel('Radius (r) [cm]')
plt.title('Comparison of $r(t_{eq})$ for Polytropic and Keplerian Cases')
plt.legend()
plt.grid(True)
plt.show()
"""