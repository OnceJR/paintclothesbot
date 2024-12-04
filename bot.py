import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os

# Inicializa MediaPipe y TensorFlow
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
model = None

# Función para cargar el modelo DeepLab (solo si se necesita segmentación avanzada)
def load_deeplab_model():
    global model
    model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False)

# Función para segmentar la imagen con MediaPipe Pose
def segment_image(image_path):
    # Leer imagen con OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar con MediaPipe Pose
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None, image  # Si no hay detección, devuelve la imagen original

    return results.pose_landmarks.landmark, image

# Función para aplicar ropa basada en segmentación
def apply_clothes(image, landmarks):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Obtener dimensiones de la imagen
    width, height = image_pil.size

    # Convertir puntos clave a píxeles
    def to_pixel(landmark):
        return int(landmark.x * width), int(landmark.y * height)

    # Obtener puntos clave necesarios
    shoulder_left = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
    shoulder_right = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    hip_left = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
    hip_right = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
    mid_hip = ((hip_left[0] + hip_right[0]) // 2, (hip_left[1] + hip_right[1]) // 2)

    # Dibujar una camiseta con forma mejorada
    shirt_points = [
        (shoulder_left[0] - 20, shoulder_left[1] - 10),  # Extender hacia afuera del hombro izquierdo
        (shoulder_right[0] + 20, shoulder_right[1] - 10),  # Extender hacia afuera del hombro derecho
        (hip_right[0] + 10, mid_hip[1] - 20),  # Cerca de la cadera derecha
        (hip_left[0] - 10, mid_hip[1] - 20),  # Cerca de la cadera izquierda
    ]
    draw.polygon(shirt_points, fill="pink", outline="black")

    # Dibujar pantalones con forma mejorada
    pants_left_leg = [
        (hip_left[0] - 5, hip_left[1]),  # Cadera izquierda
        (hip_left[0] - 15, hip_left[1] + height // 4),  # Extender hacia abajo y afuera
        (hip_left[0] + 5, hip_left[1] + height // 4),  # Extender hacia abajo y adentro
        (hip_left[0] + 5, hip_left[1]),  # Volver al punto inicial
    ]
    pants_right_leg = [
        (hip_right[0] + 5, hip_right[1]),  # Cadera derecha
        (hip_right[0] + 15, hip_right[1] + height // 4),  # Extender hacia abajo y afuera
        (hip_right[0] - 5, hip_right[1] + height // 4),  # Extender hacia abajo y adentro
        (hip_right[0] - 5, hip_right[1]),  # Volver al punto inicial
    ]

    # Separación entre las piernas (opcional, ajustar según proporción)
    leg_gap = 10
    pants_left_leg = [(x - leg_gap, y) for x, y in pants_left_leg]
    pants_right_leg = [(x + leg_gap, y) for x, y in pants_right_leg]

    # Dibujar los pantalones
    draw.polygon(pants_left_leg, fill="blue", outline="black")
    draw.polygon(pants_right_leg, fill="blue", outline="black")

    # Convertir de vuelta a OpenCV
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Manejador para el comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("¡Hola! Envíame una imagen y agregaré ropa automáticamente usando IA.")

# Manejador para las fotos
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Descargar la imagen enviada
    photo = await update.message.photo[-1].get_file()
    file_path = "temp_image.jpg"
    await photo.download_to_drive(file_path)

    # Segmentar la imagen con MediaPipe Pose
    landmarks, image = segment_image(file_path)
    if landmarks is None:
        await update.message.reply_text("No se detectó ninguna persona en la imagen.")
        os.remove(file_path)
        return

    # Aplicar ropa a la imagen segmentada
    output_image = apply_clothes(image, landmarks)
    output_path = "result_image.jpg"
    cv2.imwrite(output_path, output_image)

    # Enviar la imagen procesada al usuario
    await update.message.reply_photo(photo=open(output_path, 'rb'))

    # Limpiar archivos temporales
    os.remove(file_path)
    os.remove(output_path)

# Configuración del bot de Telegram
def main() -> None:
    application = Application.builder().token("7843171380:AAGVaxSZ4F3KjFefYmT_AKsGKih6H0xno9Y").build()

    # Registrar manejadores
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Iniciar el bot
    application.run_polling()

if __name__ == "__main__":
    main()
