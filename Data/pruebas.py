df = pd.read_csv(f'Data/data_{model}.csv')

def calculate_eta(row):
    e = row['avg_ecc']  # Usamos la excentricidad promedio ya calculada
    t_eq = row['avg_teq']
    Omega_k = row['Omega_k']
    
    def f(eta):
        return (1 / Omega_k) * (eta + e * np.sin(eta)) * (1 + e)**(-1.5) - t_eq
    
    def f_prime(eta):
        return (1 / Omega_k) * (1 + e * np.cos(eta)) * (1 + e)**(-1.5)
    
    try:
        eta_initial_guess = 0
        eta = newton(f, eta_initial_guess, fprime=f_prime)
    except RuntimeError:
        eta = np.nan  
    
    return eta

df['eta'] = df.apply(calculate_eta, axis=1)

output_path = f'Data/data_{model}.csv'
df.to_csv(output_path, index=False, float_format='%.8e')
print(f"Archivo actualizado guardado en: {output_path}")



df = pd.read_csv(f'Data/data_{model}.csv')


plt.figure(figsize=(10, 6))
plt.plot(df['avg_teq'], df['eta'], marker='o', linestyle='-', color='blue', alpha=0.7)
plt.xlabel(r'$t_{\rm eq}$ (s)', fontsize=18)
plt.ylabel(r'$\eta$', fontsize=18)
plt.title(r'$\eta(t_{\rm eq})$ ', fontsize=20)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()



def calculate_avg_z(row):
    e = row['avg_ecc']  
    r = row['r']  
    eta = row['eta']  
    a = r / (1 + e)  

    def z_func(theta):
        return a * (e + np.cos(eta)) * np.cos(theta)  

    integral, _ = quad(z_func, theta_initial, theta_final)  
    avg_z = integral / avg_norm_factor
    return avg_z

df['avg_z'] = df.apply(calculate_avg_z, axis=1)

output_path = f'Data/data_{model}.csv'
df.to_csv(output_path, index=False, float_format='%.8e')
print(f"DataFrame actualizado con avg_z guardado en: {output_path}")