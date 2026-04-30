from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename
import sys
import json
import traceback
import re

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add analyzers to path
sys.path.insert(0, BASE_DIR)

app = Flask(__name__)

# Use relative paths based on BASE_DIR
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max
app.config['SECRET_KEY'] = 'single_view_geometry_secret'

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    if not filename:
        return False
    filename = filename.strip().strip('"').strip("'")
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def clean_filename_util(filename):
    """Clean filename from unwanted characters"""
    # Remove quotes
    filename = filename.strip().strip('"').strip("'")
    # Remove any path characters
    filename = os.path.basename(filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove any non-alphanumeric except dot and underscore
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    return filename

def analyze_1point(image_path):
    """Run 1-point perspective analysis"""
    result_path = os.path.join(RESULT_FOLDER, '1point_result.png')
    
    try:
        import analyzers.vp_1point as vp1
        
        img = vp1.load_image(image_path)
        edges = vp1.preprocess(img)
        lines = vp1.detect_lines(edges)
        
        if len(lines) < 5:
            return None, f"Not enough lines detected (found {len(lines)}, need at least 5)"
        
        vp_info = vp1.find_single_vanishing_point(lines, img.shape)
        
        if vp_info:
            vp1.visualize_1point(img, lines, vp_info, result_path)
            # Return relative path for web access
            web_result_path = 'static/results/1point_result.png'
            return {
                'vanishing_points': [{'x': int(vp_info['point'][0]), 'y': int(vp_info['point'][1]), 'confidence': vp_info['num_intersections']}],
                'perspective_type': '1-Point Perspective',
                'horizon_line': None,
                'num_lines': len(lines)
            }, web_result_path
        return None, "Could not find vanishing point"
    except Exception as e:
        print(f"Error in 1-point analysis: {str(e)}")
        traceback.print_exc()
        return None, f"Error in 1-point analysis: {str(e)}"

def analyze_2point(image_path):
    """Run 2-point perspective analysis"""
    result_path = os.path.join(RESULT_FOLDER, '2point_result.png')
    
    try:
        import analyzers.vp_2point as vp2
        
        img = vp2.load_image(image_path)
        edges = vp2.preprocess(img)
        pos_lines, neg_lines = vp2.detect_lines_by_slope(edges)
        
        if len(pos_lines) < 3:
            return None, f"Not enough positive slope lines (found {len(pos_lines)}, need at least 3)"
        if len(neg_lines) < 3:
            return None, f"Not enough negative slope lines (found {len(neg_lines)}, need at least 3)"
        
        vp_left = vp2.find_vanishing_point(pos_lines, img.shape)
        vp_right = vp2.find_vanishing_point(neg_lines, img.shape)
        
        if vp_left and vp_right:
            vp_left_pt = vp_left['point']
            vp_right_pt = vp_right['point']
            horizon_pts = vp2.compute_horizon_line(vp_left_pt, vp_right_pt, img.shape)
            vp2.visualize_2point(img, pos_lines, neg_lines, vp_left_pt, vp_right_pt, horizon_pts, result_path)
            
            slope_val = (vp_right_pt[1] - vp_left_pt[1]) / (vp_right_pt[0] - vp_left_pt[0] + 1e-6)
            
            web_result_path = 'static/results/2point_result.png'
            return {
                'vanishing_points': [
                    {'x': int(vp_left_pt[0]), 'y': int(vp_left_pt[1]), 'confidence': vp_left['size'], 'label': 'Left VP'},
                    {'x': int(vp_right_pt[0]), 'y': int(vp_right_pt[1]), 'confidence': vp_right['size'], 'label': 'Right VP'}
                ],
                'perspective_type': '2-Point Perspective',
                'horizon_line': {
                    'slope': round(slope_val, 4)
                },
                'num_lines': len(pos_lines) + len(neg_lines)
            }, web_result_path
        return None, "Could not find both vanishing points"
    except Exception as e:
        print(f"Error in 2-point analysis: {str(e)}")
        traceback.print_exc()
        return None, f"Error in 2-point analysis: {str(e)}"

def analyze_3point(image_path):
    """Run 3-point perspective analysis"""
    result_path = os.path.join(RESULT_FOLDER, '3point_result.png')
    
    try:
        import analyzers.vp_3point as vp3
        
        img = vp3.load_image(image_path)
        edges = vp3.preprocess(img)
        all_lines = vp3.detect_all_lines(edges)
        
        if len(all_lines) < 10:
            return None, f"Not enough lines detected (found {len(all_lines)}, need at least 10)"
        
        left_group, right_group, vertical_group = vp3.classify_lines_by_orientation(all_lines, img.shape)
        
        vp_left = vp3.find_vanishing_point_from_group(left_group, img.shape, search_scale=4.0)
        vp_right = vp3.find_vanishing_point_from_group(right_group, img.shape, search_scale=4.0)
        vp_vertical = vp3.find_vanishing_point_from_group(vertical_group, img.shape, search_scale=3.0)
        
        vps = []
        if vp_left:
            vps.append(vp_left)
        if vp_right:
            vps.append(vp_right)
        if vp_vertical:
            vps.append(vp_vertical)
        
        web_result_path = 'static/results/3point_result.png'
        
        if len(vps) >= 3:
            vps.sort(key=lambda v: v['point'][0])
            vp3.visualize_3point(img, all_lines, vps, result_path)
            
            return {
                'vanishing_points': [
                    {'x': int(vps[0]['point'][0]), 'y': int(vps[0]['point'][1]), 'label': 'Left VP', 'confidence': vps[0].get('size', 0)},
                    {'x': int(vps[1]['point'][0]), 'y': int(vps[1]['point'][1]), 'label': 'Right VP', 'confidence': vps[1].get('size', 0)},
                    {'x': int(vps[2]['point'][0]), 'y': int(vps[2]['point'][1]), 'label': 'Vertical VP', 'confidence': vps[2].get('size', 0)}
                ],
                'perspective_type': '3-Point Perspective',
                'num_lines': len(all_lines)
            }, web_result_path
        elif len(vps) == 2:
            vp3.visualize_3point(img, all_lines, vps, result_path)
            return {
                'vanishing_points': [
                    {'x': int(vps[0]['point'][0]), 'y': int(vps[0]['point'][1]), 'confidence': vps[0].get('size', 0)},
                    {'x': int(vps[1]['point'][0]), 'y': int(vps[1]['point'][1]), 'confidence': vps[1].get('size', 0)}
                ],
                'perspective_type': '2-Point Perspective (3-point analysis attempted)',
                'num_lines': len(all_lines)
            }, web_result_path
        return None, f"Could not find sufficient vanishing points (found {len(vps)} clusters, need at least 2)"
    except Exception as e:
        print(f"Error in 3-point analysis: {str(e)}")
        traceback.print_exc()
        return None, f"Error in 3-point analysis: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("\n" + "="*50)
    print("📤 UPLOAD REQUEST RECEIVED")
    print("="*50)
    
    # Check if file exists
    if 'file' not in request.files:
        print("❌ Error: No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    perspective_type = request.form.get('perspective_type', '2')
    
    # Clean the filename using the utility function
    original_filename = file.filename
    clean_name = clean_filename_util(original_filename)
    
    print(f"📁 Original filename: {original_filename}")
    print(f"📁 Clean filename: {clean_name}")
    print(f"🎯 Perspective type: {perspective_type}")
    
    if not clean_name or clean_name == '':
        print("❌ Error: Empty filename after cleaning")
        return jsonify({'error': 'Invalid filename'}), 400
    
    # Check if file has extension
    if '.' not in clean_name:
        print(f"❌ Error: No extension in filename: {clean_name}")
        return jsonify({'error': 'File has no extension. Please use a valid image file.'}), 400
    
    # Check file extension
    if not allowed_file(clean_name):
        ext = clean_name.rsplit('.', 1)[1].lower() if '.' in clean_name else 'unknown'
        print(f"❌ Error: Extension '{ext}' not allowed")
        return jsonify({'error': f'File type .{ext} not allowed. Allowed: png, jpg, jpeg, bmp, tiff'}), 400
    
    # Save uploaded file
    try:
        filename = secure_filename(clean_name)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"✅ File saved to: {filepath}")
    except Exception as e:
        print(f"❌ Error saving file: {str(e)}")
        return jsonify({'error': f'Error saving file: {str(e)}'}), 500
    
    # Analyze based on perspective type
    print(f"\n🔍 Starting {perspective_type}-point analysis...")
    
    try:
        if perspective_type == '1':
            result, output_path = analyze_1point(filepath)
        elif perspective_type == '2':
            result, output_path = analyze_2point(filepath)
        elif perspective_type == '3':
            result, output_path = analyze_3point(filepath)
        else:
            print(f"❌ Invalid perspective type: {perspective_type}")
            return jsonify({'error': 'Invalid perspective type. Use 1, 2, or 3'}), 400
        
        if result is None:
            print(f"❌ Analysis failed: {output_path}")
            return jsonify({'error': output_path}), 400
        
        print(f"✅ Analysis successful!")
        print(f"📊 Result: {result.get('perspective_type')}")
        print(f"📍 Found {len(result.get('vanishing_points', []))} vanishing points")
        print("="*50)
        
        # Return results
        return jsonify({
            'success': True,
            'result_image': output_path,
            'data': result
        })
    except Exception as e:
        print(f"❌ Unexpected error during analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    file_path = os.path.join(BASE_DIR, 'static', path)
    return send_file(file_path)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎯 SINGLE VIEW GEOMETRY WEB APP")
    print("="*50)
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Results folder: {RESULT_FOLDER}")
    print("="*50)
    print("Server running at: http://127.0.0.1:5000")
    print("Press CTRL+C to stop")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, threaded=True)
