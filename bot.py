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

    # Función para validar puntos clave y asignar valores predeterminados si no son válidos
    def get_valid_landmark(landmark, default_x, default_y):
        if landmark.visibility < 0.5:  # Umbral de visibilidad
            return default_x, default_y
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return max(0, min(x, width - 1)), max(0, min(y, height - 1))

    # Obtener puntos clave y asignar valores predeterminados si son inválidos
    shoulder_left = get_valid_landmark(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], width // 4, height // 3)
    shoulder_right = get_valid_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], (width * 3) // 4, height // 3)
    hip_left = get_valid_landmark(landmarks[mp_pose.PoseLandmark.LEFT_HIP], width // 4, (height * 2) // 3)
    hip_right = get_valid_landmark(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], (width * 3) // 4, (height * 2) // 3)
    mid_hip = ((hip_left[0] + hip_right[0]) // 2, (hip_left[1] + hip_right[1]) // 2)

    # Dibujar una camiseta con forma realista (polígono)
    shirt_points = [
        (shoulder_left[0] - 10, shoulder_left[1]),  # Punto superior izquierdo
        (shoulder_right[0] + 10, shoulder_right[1]),  # Punto superior derecho
        (hip_right[0] + 15, mid_hip[1] - 20),  # Punto inferior derecho
        (hip_left[0] - 15, mid_hip[1] - 20),  # Punto inferior izquierdo
    ]
    draw.polygon(shirt_points, fill="pink", outline="black")

    # Dibujar pantalones con forma realista (dos polígonos para las piernas)
    pants_left_leg = [
        (hip_left[0] - 10, hip_left[1]),
        (mid_hip[0] - 5, mid_hip[1] + 10),
        (mid_hip[0] - 20, mid_hip[1] + height // 6),
        (hip_left[0] - 25, hip_left[1] + height // 6),
    ]
    pants_right_leg = [
        (hip_right[0] + 10, hip_right[1]),
        (mid_hip[0] + 5, mid_hip[1] + 10),
        (mid_hip[0] + 20, mid_hip[1] + height // 6),
        (hip_right[0] + 25, hip_right[1] + height // 6),
    ]

    draw.polygon(pants_left_leg, fill="blue", outline="black")
    draw.polygon(pants_right_leg, fill="blue", outline="black")

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
