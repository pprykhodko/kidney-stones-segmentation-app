import os
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from PIL import Image

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


@app.route('/')
def home():
    return render_template('index.html')


@app.post('/segment')
def upload_file():
    file = request.files.get('image')

    if file is None or not file.filename:
        return render_template(
            'index.html',
            error='Please choose an image.'
        ), 400

    extension = file.filename.rsplit('.', 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return render_template(
            'index.html',
            error='Please upload a JPG or PNG image.'
        ), 400

    try:
        with Image.open(file.stream) as image:
            if image.format not in {'JPEG', 'PNG'}:
                return render_template(
                    'index.html',
                    error='Please upload a JPG or PNG image.'
                ), 400

            extension = 'jpg' if image.format == 'JPEG' else 'png'
            image.verify()
    except (OSError, ValueError, Image.DecompressionBombError):
        return render_template(
            'index.html',
            error='This file is not a valid image.'
        ), 400

    file.stream.seek(0)
    filename = f'{uuid4().hex}.{extension}'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    upload_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_image_path)

    uploaded_image = url_for('static', filename=f'uploads/{filename}')

    return render_template(
        'index.html',
        uploaded_image=uploaded_image
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        'index.html',
        error='The image must be smaller than 10 MB.'
    ), 413


if __name__ == '__main__':
    app.run()
