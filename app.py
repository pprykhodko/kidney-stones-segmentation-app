import os
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from PIL import Image

from segmentator import segment_image

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(app.static_folder, 'results')
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

    try:
        mask, overlay = segment_image(upload_image_path)
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        image_id = os.path.splitext(filename)[0]
        mask_filename = f'{image_id}_mask.png'
        overlay_filename = f'{image_id}_overlay.png'
        Image.fromarray(mask).save(
            os.path.join(app.config['RESULTS_FOLDER'], mask_filename)
        )
        Image.fromarray(overlay).save(
            os.path.join(app.config['RESULTS_FOLDER'], overlay_filename)
        )
    except (OSError, ValueError, RuntimeError):
        app.logger.exception('Image segmentation failed')

        return render_template(
            'index.html',
            uploaded_image=uploaded_image,
            error='Unable to segment this image. Please try again.'
        ), 500

    return render_template(
        'index.html',
        uploaded_image=uploaded_image,
        mask_image=url_for('static', filename=f'results/{mask_filename}'),
        segmented_image=url_for('static', filename=f'results/{overlay_filename}')
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        'index.html',
        error='The image must be smaller than 10 MB.'
    ), 413


if __name__ == '__main__':
    app.run()
