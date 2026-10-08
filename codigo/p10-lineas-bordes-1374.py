import cv2
import numpy as np
# Cristopher Lopez NC 1374
# Cargar imagen
imagen = cv2.imread("imagenes/serpiente.jpg")

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32
gris_float = np.float32(gris)

# Detectar esquinas mediante Harris
esquinas = cv2.cornerHarris(
    gris_float,
    2,
    3,
    0.04
)

# Dilatar para hacer visibles las esquinas
esquinas = cv2.dilate(
    esquinas,
    None
)

# Crear copia
resultado = imagen.copy()

# Umbral para identificar esquinas
umbral = 0.05 * esquinas.max()

# Marcar esquinas
resultado[esquinas > umbral] = [0, 0, 255]

# Mostrar resultados
cv2.imshow(
    "Imagen original.jpg",
    imagen
)
cv2.imshow(
    "experimento.jpg",
    imagen
)
cv2.imshow(
    "Esquinas detectadas",
    resultado
)
cv2.imshow(
    "experimento.jpg",
    resultado
)
# Guardar resultado
cv2.imwrite(
    "resultados/Experimento de Harris.jpg",
    resultado
)

# Contar esquinas aproximadas
cantidad_esquinas = np.sum(
    esquinas > umbral
)

print("Deteccion de esquinas terminada.")
print("Cantidad aproximada de puntos detectados:",
      cantidad_esquinas)

print("Resultado guardado en:")
print("resultados/Experimento de Harris")

# Esperar una tecla
cv2.waitKey(0)


# Cerrar ventanas
cv2.destroyAllWindows()

print("Cristopher Lopez NC 1374")