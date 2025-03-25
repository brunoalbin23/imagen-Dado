import cv2
from PIL import Image
from rembg import remove
import numpy as np
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_FOLDER = os.path.join(BASE_PATH, "img")
OUTPUT_FOLDER = os.path.join(BASE_PATH, "procesadas")

# Crear la carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Definir las rutas de los dados
DICE_IMAGES = {i: os.path.join(BASE_PATH, "dados", f"dado_{i}n.png") for i in range(1, 7)}

# Funciones de procesamiento
def convertir_a_grises(imagen):
    if imagen is None:
        return None  # Retorna None si la imagen no se cargó correctamente
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

def sacar_fondo(imagen):
    if imagen is None:
        return None
    return cv2.imdecode(np.frombuffer(remove(cv2.imencode(".png", imagen)[1].tobytes()), np.uint8), cv2.IMREAD_UNCHANGED)

def redimensionar_cuadrado(imagen, tamano_final=(64, 64)):
    alto, ancho = imagen.shape[:2]
    max_lado = max(alto, ancho)

    # Crear una imagen en blanco del tamaño del lado más grande
    if len(imagen.shape) == 2:  # Imagen en escala de grises
        imagen_cuadrada = np.ones((max_lado, max_lado), dtype=np.uint8) * 255
    else:  # Imagen con canales de color o transparencia
        imagen_cuadrada = np.ones((max_lado, max_lado, imagen.shape[2]), dtype=np.uint8) * 255

    # Centrar la imagen original dentro de la nueva imagen cuadrada
    y_offset = (max_lado - alto) // 2
    x_offset = (max_lado - ancho) // 2
    imagen_cuadrada[y_offset:y_offset + alto, x_offset:x_offset + ancho] = imagen

    # Redimensionar 
    return cv2.resize(imagen_cuadrada, tamano_final, interpolation=cv2.INTER_LINEAR)

def reducir_a_6_tonos(imagen):
    if imagen is None:
        return None

    if imagen.shape[-1] == 4:  # Si tiene canal alfa (transparencia)
        alpha = imagen[:, :, 3]
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
        imagen[alpha == 0] = 255  # Fondo blanco en áreas transparentes
    else:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    niveles_gris = np.array([255, 204, 153, 102, 51, 0], dtype=np.uint8)
    indices = np.digitize(imagen, bins=niveles_gris, right=True)
    indices = np.clip(indices, 1, 6)
    return niveles_gris[indices - 1]

def imagen_a_dados(imagen, tamano_dado=16):
    if imagen is None:
        return None
    
    alto, ancho = imagen.shape[:2]
    imagen_salida = Image.new("RGBA", (ancho * tamano_dado, alto * tamano_dado), (255, 255, 255, 255))
    dados_redimensionados = {i: Image.open(DICE_IMAGES[i]).convert("RGBA").resize((tamano_dado, tamano_dado), Image.LANCZOS) for i in range(1, 7)}
    
    for y in range(alto):
        for x in range(ancho):
            gris = imagen[y, x]
            nivel = np.digitize(gris, np.array([255, 204, 153, 102, 51, 0]), right=True)
            nivel = np.clip(nivel, 1, 6)
            imagen_salida.paste(dados_redimensionados[nivel], (x * tamano_dado, y * tamano_dado), mask=dados_redimensionados[nivel])
    
    return imagen_salida

# Procesar todas las imágenes en la carpeta img (comentar esto si queremos solo la última imagen)
for archivo in os.listdir(IMG_FOLDER):
    if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
        ruta_original = os.path.join(IMG_FOLDER, archivo)
        nombre_base, _ = os.path.splitext(archivo)
        carpeta_salida = os.path.join(OUTPUT_FOLDER, nombre_base)
        os.makedirs(carpeta_salida, exist_ok=True)
        
        imagen = cv2.imread(ruta_original, cv2.IMREAD_UNCHANGED)
        
        if imagen is None:
            print(f"Error al cargar la imagen {archivo}, saltando procesamiento.")
            continue  # Evita errores al intentar procesar una imagen inexistente
        
        # Paso 1: Imagen en escala de grises
        img_gris = convertir_a_grises(imagen)
        if img_gris is not None:
            cv2.imwrite(os.path.join(carpeta_salida, "1_gris.png"), img_gris)
        
        # Paso 2: Sacar fondo
        img_sin_fondo = sacar_fondo(imagen)
        if img_sin_fondo is not None:
            cv2.imwrite(os.path.join(carpeta_salida, "2_sin_fondo.png"), img_sin_fondo)
        
        # Paso 3: Agrandar pixeles
        img_pixeleada = redimensionar_cuadrado(img_sin_fondo)
        if img_pixeleada is not None:
            cv2.imwrite(os.path.join(carpeta_salida, "3_pixeleada.png"), img_pixeleada)
        
        # Paso 4: Reducir a 6 tonos
        img_6_tonos = reducir_a_6_tonos(img_pixeleada)
        if img_6_tonos is not None:
            cv2.imwrite(os.path.join(carpeta_salida, "4_6_tonos.png"), img_6_tonos)
        
        # Paso 5: Convertir a dados
        img_dados = imagen_a_dados(img_6_tonos)
        if img_dados is not None:
            img_dados.save(os.path.join(carpeta_salida, "5_dados.png"))
        
        print(f"Procesamiento completado para {archivo}")

# SOLO PARA GENERAR LA ULTIMA IMAGEN SIN PASOS INTERMEDIOS (descomentar esto si queremos solo una imagen)
"""  for archivo in os.listdir(IMG_FOLDER):
     if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
         ruta_original = os.path.join(IMG_FOLDER, archivo)
         nombre_base, _ = os.path.splitext(archivo)
        imagen = cv2.imread(ruta_original, cv2.IMREAD_UNCHANGED)
         
         if imagen is None:
             print(f"Error al cargar la imagen {archivo}, saltando procesamiento.")
             continue
         
         img_gris = convertir_a_grises(imagen)
         img_sin_fondo = sacar_fondo(imagen)
         img_pixeleada = redimensionar_cuadrado(img_sin_fondo)
         img_6_tonos = reducir_a_6_tonos(img_pixeleada)
         img_dados = imagen_a_dados(img_6_tonos)
         
         if img_dados is not None:
             img_dados.save(os.path.join(OUTPUT_FOLDER, f"{nombre_base}_final.png"))
         
         print(f"Última imagen generada para {archivo}")"
"""