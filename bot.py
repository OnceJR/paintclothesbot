import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os

# Cargar modelo DeepLab preentrenado
def load_deeplab_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False)
    return model

# Procesar imagen con DeepLab
def segment_image(image_path, model):
    # Cargar imagen y cambiar tamaño
    img = cv2.imread(image_path)
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, [513, 513])
    img_tensor = tf.expand_dims(img_tensor / 127.5 - 1.0, axis=0)

    # Realizar predicción
    output = model.predict(img_tensor)
    segmentation_map = tf.argmax(output[0], axis=-1).numpy()

    # Restaurar el tamaño original
    segmentation_map_resized = cv2.resize(segmentation_map, original_size[::-1], interpolation=cv2.INTER_NEAREST)
    return segmentation_map_resized

# Aplicar ropa basada en segmentación
def apply_clothes(image_path, segmentation_map):
    # Cargar imagen original
    img = Image.open(image_path).convert("RGBA")
    img_draw = ImageDraw.Draw(img, "RGBA")

    # Definir colores para la ropa
    torso_color = (255, 20, 147, 128)  # Camiseta rosa
    leg_color = (30, 144, 255, 128)   # Pantalones azules

    # Identificar regiones segmentadas
    torso_mask = segmentation_map == 15  # Suponiendo que 15 es el torso
    leg_mask = segmentation_map == 14    # Suponiendo que 14 son las piernas

    # Aplicar color al torso
    if np.any(torso_mask):
        img_draw.rectangle([(50, 50), (200, 300)], fill=torso_color)

    # Aplicar color a las piernas
    if np.any(leg_mask):
        img_draw.rectangle([(100, 300), (150, 500)], fill=leg_color)

    return img

# Manejador para el comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("¡Hola! Envíame una imagen y agregaré ropa automáticamente usando DeepLab.")

# Manejador para las fotos
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Descargar la imagen enviada
    photo = await update.message.photo[-1].get_file()
    file_path = "temp_image.jpg"
    await photo.download_to_drive(file_path)

    # Cargar el modelo DeepLab
    model = load_deeplab_model()

    # Segmentar la imagen
    segmentation_map = segment_image(file_path, model)

    # Aplicar ropa a la imagen segmentada
    output_image = apply_clothes(file_path, segmentation_map)
    output_path = "result_image.png"
    output_image.save(output_path)

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
