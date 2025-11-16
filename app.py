#!/usr/bin/env python3
"""
Flask app for interactive image effects workshop.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import base64
from io import BytesIO
from image_effects import ImageEffects

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create folders if they don't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get image dimensions
        effects = ImageEffects(filepath)
        width, height = effects.image.size

        return jsonify({
            'success': True,
            'filename': filename,
            'width': width,
            'height': height
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/apply_effect', methods=['POST'])
def apply_effect():
    data = request.json
    filename = data.get('filename')
    effect = data.get('effect')
    params = data.get('params', {})

    if not filename or not effect:
        return jsonify({'error': 'Missing filename or effect'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        effects = ImageEffects(filepath)

        if effect == 'paint_stroke':
            result = effects.paint_stroke_effect(
                stroke_size=params.get('stroke_size', 8),
                stroke_length=params.get('stroke_length', 12),
                stroke_width=params.get('stroke_width', 3),
                angle_variation=params.get('angle_variation', 30),
                density=params.get('density', 1.0)
            )
        elif effect == 'japanese_calligraphy':
            result = effects.japanese_calligraphy_effect(
                brush_size=params.get('brush_size', 12),
                stroke_length=params.get('stroke_length', 25),
                tail_taper=params.get('tail_taper', 0.7),
                edge_influence=params.get('edge_influence', 0.7),
                density=params.get('density', 1.2),
                follow_edges=params.get('follow_edges', True),
                ink_bleed=params.get('ink_bleed', True),
                smart_tree_mode=params.get('smart_tree_mode', False),
                directional_mode=params.get('directional_mode', False),
                directional_regions=params.get('directional_regions', []),
                variable_brush_size=params.get('variable_brush_size', False),
                min_brush_size=params.get('min_brush_size', 6),
                max_brush_size=params.get('max_brush_size', 18),
                size_variation=params.get('size_variation', 0.7)
            )
        elif effect == 'sumi_e_brush':
            result = effects.sumi_e_brush_effect(
                brush_size=params.get('brush_size', 15),
                min_stroke_length=params.get('min_stroke_length', 20),
                max_stroke_length=params.get('max_stroke_length', 60),
                pressure_variation=params.get('pressure_variation', 0.8),
                texture_intensity=params.get('texture_intensity', 0.6),
                ink_dispersion=params.get('ink_dispersion', 0.4),
                flow_coherence=params.get('flow_coherence', 0.75),
                paper_texture=params.get('paper_texture', True)
            )
        elif effect == 'oil_paint':
            result = effects.oil_paint_effect(
                min_brush_size=params.get('min_brush_size', 8),
                max_brush_size=params.get('max_brush_size', 25),
                stroke_length_mult=params.get('stroke_length_mult', 2.5),
                impasto_strength=params.get('impasto_strength', 0.7),
                color_blend=params.get('color_blend', 0.5),
                detail_sensitivity=params.get('detail_sensitivity', 0.6),
                stroke_direction_coherence=params.get('stroke_direction_coherence', 0.8),
                paint_thickness=params.get('paint_thickness', 0.6),
                fast_mode=params.get('fast_mode', False)
            )
        elif effect == 'linocut':
            result = effects.linocut_effect(
                num_colors=params.get('num_colors', 4),
                color_boost=params.get('color_boost', 1.3),
                outline_thickness=params.get('outline_thickness', 2),
                texture_intensity=params.get('texture_intensity', 0.6),
                carve_direction=params.get('carve_direction', 'mixed'),
                contrast=params.get('contrast', 1.2)
            )
        else:
            return jsonify({'error': 'Unknown effect'}), 400

        # Convert to base64 for preview
        img_base64 = image_to_base64(result)

        return jsonify({
            'success': True,
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download', methods=['POST'])
def download_image():
    data = request.json
    filename = data.get('filename')
    effect = data.get('effect')
    params = data.get('params', {})

    if not filename or not effect:
        return jsonify({'error': 'Missing filename or effect'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        effects = ImageEffects(filepath)

        if effect == 'paint_stroke':
            result = effects.paint_stroke_effect(
                stroke_size=params.get('stroke_size', 8),
                stroke_length=params.get('stroke_length', 12),
                stroke_width=params.get('stroke_width', 3),
                angle_variation=params.get('angle_variation', 30),
                density=params.get('density', 1.0)
            )
        elif effect == 'japanese_calligraphy':
            result = effects.japanese_calligraphy_effect(
                brush_size=params.get('brush_size', 12),
                stroke_length=params.get('stroke_length', 25),
                tail_taper=params.get('tail_taper', 0.7),
                edge_influence=params.get('edge_influence', 0.7),
                density=params.get('density', 1.2),
                follow_edges=params.get('follow_edges', True),
                ink_bleed=params.get('ink_bleed', True),
                smart_tree_mode=params.get('smart_tree_mode', False),
                directional_mode=params.get('directional_mode', False),
                directional_regions=params.get('directional_regions', []),
                variable_brush_size=params.get('variable_brush_size', False),
                min_brush_size=params.get('min_brush_size', 6),
                max_brush_size=params.get('max_brush_size', 18),
                size_variation=params.get('size_variation', 0.7)
            )
        elif effect == 'sumi_e_brush':
            result = effects.sumi_e_brush_effect(
                brush_size=params.get('brush_size', 15),
                min_stroke_length=params.get('min_stroke_length', 20),
                max_stroke_length=params.get('max_stroke_length', 60),
                pressure_variation=params.get('pressure_variation', 0.8),
                texture_intensity=params.get('texture_intensity', 0.6),
                ink_dispersion=params.get('ink_dispersion', 0.4),
                flow_coherence=params.get('flow_coherence', 0.75),
                paper_texture=params.get('paper_texture', True)
            )
        elif effect == 'oil_paint':
            result = effects.oil_paint_effect(
                min_brush_size=params.get('min_brush_size', 8),
                max_brush_size=params.get('max_brush_size', 25),
                stroke_length_mult=params.get('stroke_length_mult', 2.5),
                impasto_strength=params.get('impasto_strength', 0.7),
                color_blend=params.get('color_blend', 0.5),
                detail_sensitivity=params.get('detail_sensitivity', 0.6),
                stroke_direction_coherence=params.get('stroke_direction_coherence', 0.8),
                paint_thickness=params.get('paint_thickness', 0.6),
                fast_mode=params.get('fast_mode', False)
            )
        elif effect == 'linocut':
            result = effects.linocut_effect(
                num_colors=params.get('num_colors', 4),
                color_boost=params.get('color_boost', 1.3),
                outline_thickness=params.get('outline_thickness', 2),
                texture_intensity=params.get('texture_intensity', 0.6),
                carve_direction=params.get('carve_direction', 'mixed'),
                contrast=params.get('contrast', 1.2)
            )
        else:
            return jsonify({'error': 'Unknown effect'}), 400

        # Save result
        result_filename = f"{Path(filename).stem}_{effect}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        result.save(result_path, quality=95)

        return jsonify({
            'success': True,
            'download_url': f'/results/{result_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


if __name__ == '__main__':
    print("\nImage Effects Workshop")
    print("=" * 50)
    print("Starting server at http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
