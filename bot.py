import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import os

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("¡Hola! Envíame una imagen y agregaré ropa automáticamente.")

def handle_photo(update: Update, context: CallbackContext) -> None:
    file = update.message.photo[-1].get_file()
    file_path = file.download("temp_image.jpg")

    # Leer imagen con OpenCV
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar con MediaPipe para detectar pose
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        update.message.reply_text("No se detectó ninguna persona en la imagen. Intenta otra.")
        os.remove(file_path)
        return

    # Dibujar ropa en base a los puntos clave
    annotated_image = draw_clothes(image, results.pose_landmarks.landmark)

    # Guardar imagen procesada
    edited_path = "edited_image.jpg"
    cv2.imwrite(edited_path, annotated_image)

    # Enviar la imagen de vuelta al usuario
    update.message.reply_photo(photo=open(edited_path, 'rb'))

    # Eliminar archivos temporales
    os.remove(file_path)
    os.remove(edited_path)

def draw_clothes(image, landmarks):
    # Convertir OpenCV a PIL para dibujar
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Obtener dimensiones de la imagen
    width, height = image_pil.size

    # Extraer puntos clave necesarios
    shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Convertir coordenadas normalizadas a píxeles
    def to_pixel(landmark):
        return int(landmark.x * width), int(landmark.y * height)

    # Dibujar una camiseta
    top_left = to_pixel(shoulder_left)
    top_right = to_pixel(shoulder_right)
    bottom_left = to_pixel(hip_left)
    bottom_right = to_pixel(hip_right)

    draw.rectangle([top_left, bottom_right], fill="pink", outline="black")

    # Dibujar pantalones
    pants_top_left = bottom_left
    pants_top_right = bottom_right
    pants_bottom_left = (pants_top_left[0], pants_top_left[1] + 100)
    pants_bottom_right = (pants_top_right[0], pants_top_right[1] + 100)

    draw.rectangle([pants_top_left, pants_bottom_right], fill="blue", outline="black")

    # Convertir de vuelta a OpenCV
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def main() -> None:
    # Crea la aplicación del bot
    application = Application.builder().token("7843171380:AAGVaxSZ4F3KjFefYmT_AKsGKih6H0xno9Y").build()

    # Agregar manejadores de comandos y mensajes
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Iniciar el bot
    application.run_polling()

if __name__ == "__main__":
    main()
