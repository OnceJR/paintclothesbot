import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("¡Hola! Envíame una imagen y agregaré ropa automáticamente.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Obtén el archivo
    file = await context.bot.get_file(update.message.photo[-1].file_id)
    file_path = "temp_image.jpg"
    
    # Descarga el archivo al disco
    await file.download_to_drive(file_path)

    # Leer imagen con OpenCV
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar con MediaPipe para detectar pose
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        await update.message.reply_text("No se detectó ninguna persona en la imagen. Intenta otra.")
        os.remove(file_path)
        return

    # Dibujar ropa en base a los puntos clave
    try:
        annotated_image = draw_clothes(image, results.pose_landmarks.landmark)
    except ValueError as e:
        await update.message.reply_text(f"Error al procesar la imagen: {e}")
        os.remove(file_path)
        return

    # Guardar imagen procesada
    edited_path = "edited_image.jpg"
    cv2.imwrite(edited_path, annotated_image)

    # Enviar la imagen de vuelta al usuario
    await update.message.reply_photo(photo=open(edited_path, 'rb'))

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
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        # Limitar las coordenadas al rango de la imagen
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return x, y

    # Convertir puntos clave a píxeles
    top_left = to_pixel(shoulder_left)
    top_right = to_pixel(shoulder_right)
    bottom_left = to_pixel(hip_left)
    bottom_right = to_pixel(hip_right)

    # Verificar que las coordenadas son válidas para un rectángulo
    if top_left[0] >= bottom_right[0] or top_left[1] >= bottom_right[1]:
        raise ValueError("Coordenadas inválidas para dibujar la ropa.")

    # Dibujar una camiseta
    draw.rectangle([top_left, bottom_right], fill="pink", outline="black")

    # Dibujar pantalones
    pants_top_left = bottom_left
    pants_top_right = bottom_right
    pants_bottom_left = (pants_top_left[0], pants_top_left[1] + 100)
    pants_bottom_right = (pants_top_right[0], pants_top_right[1] + 100)

    # Asegurarse de que las coordenadas del pantalón son válidas
    if pants_bottom_left[1] > height or pants_bottom_right[1] > height:
        raise ValueError("Coordenadas inválidas para los pantalones.")

    draw.rectangle([pants_top_left, pants_bottom_right], fill="blue", outline="black")

    # Convertir de vuelta a OpenCV
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def main() -> None:
    # Crea la aplicación del bot con tu token
    application = Application.builder().token("7843171380:AAGVaxSZ4F3KjFefYmT_AKsGKih6H0xno9Y").build()

    # Agregar manejadores de comandos y mensajes
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Iniciar el bot
    application.run_polling()

if __name__ == "__main__":
    main()
