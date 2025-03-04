#!/bin/bash

# Script para generar gráficos combinados variando la masa

# Ruta al script de Python
PYTHON_SCRIPT="generate_LSO_plots.py"

# Rango de masas (de 20 a 100 en pasos de 5)
MASS_VALUES=$(seq 20 5 100)

# Valor fijo de eta_D
ETA_VALUE=0.5

# Directorio base dinámico basado en eta_D
BASE_PATH_PREFIX="/fs/phaethon/other0/gibran/sim_results/Si28_5d-2_W9_D"
BASE_PATH="${BASE_PATH_PREFIX}${ETA_VALUE}"

# Directorio de salida para los gráficos
OUTPUT_PATH="/fs/phaethon/other0/gibran/accretion/plots"

# Crear directorio de salida si no existe
mkdir -p "$OUTPUT_PATH"

# Iterar sobre las masas
for MASS in $MASS_VALUES; do
    echo "Generando gráfico para masa=${MASS}M_sun y eta_D=${ETA_VALUE}..."

    # Construir la ruta base de logs
    LOGS_DIR="${BASE_PATH}/${MASS}M_W0.9_D${ETA_VALUE}"

    # Verificar si el directorio existe antes de ejecutar el script
    if [ -d "$LOGS_DIR" ]; then
        # Ejecutar el script de Python con los parámetros dinámicos
        python "$PYTHON_SCRIPT" --mass "$MASS" --eta_d "$ETA_VALUE"

        # Verificar si el gráfico se generó correctamente
        MODEL="${MASS}M_W0.6_D${ETA_VALUE}"
        PLOT_PATH="../plots/Combined_LSO_${MODEL}.pdf"
        if [ -f "$PLOT_PATH" ]; then
            echo "Gráfico generado: $PLOT_PATH"
        else
            echo "Error: No se generó el gráfico para masa=${MASS}M_sun y eta_D=${ETA_VALUE}"
        fi
    else
        echo "Advertencia: No se encontró el directorio ${LOGS_DIR}. Saltando esta masa."
    fi
done

echo "Proceso completo."

