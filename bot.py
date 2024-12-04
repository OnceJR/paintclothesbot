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

    # Extraer puntos clave necesarios y asignar valores predeterminados si no son válidos
    def get_valid_landmark(landmark, default_x, default_y):
        if landmark.visibility < 0.5:  # Umbral de visibilidad
            return default_x, default_y
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return max(0, min(x, width - 1)), max(0, min(y, height - 1))

    # Obtener puntos clave con valores predeterminados
    shoulder_left = get_valid_landmark(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], width // 4, height // 3)
    shoulder_right = get_valid_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], (width * 3) // 4, height // 3)
    hip_left = get_valid_landmark(landmarks[mp_pose.PoseLandmark.LEFT_HIP], width // 4, (height * 2) // 3)
    hip_right = get_valid_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], (width * 3) // 4, (height * 2) // 3)

    # Asegurar que las coordenadas no estén invertidas
    def sort_coordinates(coord1, coord2):
        x0, y0 = coord1
        x1, y1 = coord2
        return (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))

    # Rectángulo de la camiseta
    shirt_top_left, shirt_bottom_right = sort_coordinates(shoulder_left, hip_right)
    draw.rectangle([shirt_top_left, shirt_bottom_right], fill="pink", outline="black")

    # Rectángulo de los pantalones
    pants_top_left, pants_bottom_right = sort_coordinates(
        hip_left, (hip_right[0], hip_right[1] + height // 6)
    )
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
