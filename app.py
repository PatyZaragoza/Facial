from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import cv2
import dlib  # Biblioteca para detección de puntos faciales
import base64

app = Flask(__name__)

# Cargar el predictor de forma facial de dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"

try:
    predictor = dlib.shape_predictor(predictor_path)
except RuntimeError as e:
    print(f"Error al cargar el predictor: {e}")
    exit(1)  # Salir del programa si no se puede cargar el predictor

detector = dlib.get_frontal_face_detector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images')
def list_images():
    # Obtener la lista de imágenes en la carpeta 'imagenes'
    imagenes_dir = 'imagenes'
    images = [img for img in os.listdir(imagenes_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    return jsonify(images)

@app.route('/upload', methods=['POST'])
def upload_image():
    # Ahora aceptamos el nombre de la imagen y la cargamos desde la carpeta
    image_name = request.form.get('image')
    if not image_name:
        return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400

    # Abrir la imagen desde la carpeta 'imagenes'
    image_path = os.path.join('imagenes', image_name)
    try:
        # Abrir la imagen con PIL
        image = Image.open(image_path)
        # El resto de tu código se mantiene igual...

        image_np = np.array(image)

        # Convertir la imagen a escala de grises para detección de rostros
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = detector(gray)

        # Crear un objeto de dibujo para agregar las "X"
        draw = ImageDraw.Draw(image)

        # Dibujar puntos faciales en la imagen
        for face in faces:
            landmarks = predictor(gray, face)

            # Índices de los puntos de interés
            puntos_a_dibujar = [
                21, 22,  # Extremos de las cejas
                17, 25,
                36, 37, 38,  # Ojo izquierdo (esquinas y centro)
                42, 43, 44,  # Ojo derecho (esquinas y centro)
                30,  # Nariz
                51,  # Labio superior (centro)
                57,  # Labio inferior (centro)
                48, 54  # Labios (lados)
            ]

            for i in puntos_a_dibujar:
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                # Dibujar una "X" en cada punto
                # Ajustar el tamaño de la "X" aquí
                draw.line((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0), width=1)  # Línea diagonal de la X
                draw.line((x - 2, y + 2, x + 2, y - 2), fill=(255, 0, 0), width=1)  # Otra línea diagonal de la X

            # Ajustar los puntos de los ojos
            # Ojo izquierdo
            eye_left_x_center = (landmarks.part(36).x + landmarks.part(39).x) // 2
            eye_left_y = landmarks.part(37).y
            draw.line((eye_left_x_center - 2, eye_left_y - 3, eye_left_x_center + 2, eye_left_y + 3), fill=(255, 0, 0), width=1)
            draw.line((eye_left_x_center - 2, eye_left_y + 3, eye_left_x_center + 2, eye_left_y - 3), fill=(255, 0, 0), width=1)

            # Ojo derecho
            eye_right_x_center = (landmarks.part(42).x + landmarks.part(45).x) // 2
            eye_right_y = landmarks.part(43).y
            draw.line((eye_right_x_center - 2, eye_right_y - 3, eye_right_x_center + 2, eye_right_y + 3), fill=(255, 0, 0), width=1)
            draw.line((eye_right_x_center - 2, eye_right_y + 3, eye_right_x_center + 2, eye_right_y - 3), fill=(255, 0, 0), width=1)

        # Convertir la imagen con detección de puntos faciales de vuelta a formato PIL para mostrar
        image_with_landmarks = image

        # Guardar la imagen con los puntos faciales en un buffer
        buf = io.BytesIO()
        image_with_landmarks.save(buf, format='JPEG')
        buf.seek(0)
        image_data = buf.read()

        # Devolver la imagen codificada en base64 para mostrar en el navegador
        image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(image_data).decode('utf-8')

        # Enviar la cantidad de rostros detectados y la imagen procesada
        return jsonify({'matrix': f'Rostros detectados: {len(faces)}', 'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
