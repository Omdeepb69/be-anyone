import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image 
from insightface.model_zoo import get_model

def create_output_folder(folder_name="faces"):
    """Create output folder if it doesn't exist"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name

def load_image(image_path):
    """Load and validate image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return img

def swap_faces(swapper, app, source_img, target_img, source_face_idx=0, target_face_idx=0):
    """
    Swap faces between two images
    
    Args:
        swapper: InsightFace swapper model
        app: FaceAnalysis app
        source_img: Source image (face to be copied)
        target_img: Target image (face to be replaced)
        source_face_idx: Index of face in source image (default: 0 for first face)
        target_face_idx: Index of face in target image (default: 0 for first face)
    
    Returns:
        result_img: Image with swapped face
    """
    
    # Get faces from both images
    source_faces = app.get(source_img)
    target_faces = app.get(target_img)
    
    if len(source_faces) == 0:
        raise ValueError("No faces found in source image!")
    
    if len(target_faces) == 0:
        raise ValueError("No faces found in target image!")
    
    # Select faces based on indices
    if source_face_idx >= len(source_faces):
        print(f"Source face index {source_face_idx} out of range. Using index 0.")
        source_face_idx = 0
    
    if target_face_idx >= len(target_faces):
        print(f"Target face index {target_face_idx} out of range. Using index 0.")
        target_face_idx = 0
    
    source_face = source_faces[source_face_idx]
    target_face = target_faces[target_face_idx]
    
    # Perform face swap
    result_img = swapper.get(target_img, target_face, source_face, paste_back=True)
    
    return result_img

def save_result(img, output_path):
    """Save image to specified path"""
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"Saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")
    return success

def get_filename_without_extension(filepath):
    """Get filename without extension from filepath"""
    return os.path.splitext(os.path.basename(filepath))[0]

def main():
    parser = argparse.ArgumentParser(description='Face Swapper using InsightFace')
    parser.add_argument('--source', '-s', required=True, help='Path to source image (face to be copied)')
    parser.add_argument('--target', '-t', required=True, help='Path to target image (face to be replaced)')
    parser.add_argument('--output_folder', '-o', default='faces', help='Output folder name (default: faces)')
    parser.add_argument('--source_face_idx', type=int, default=0, help='Index of face in source image (default: 0)')
    parser.add_argument('--target_face_idx', type=int, default=0, help='Index of face in target image (default: 0)')
    parser.add_argument('--both_ways', '-b', action='store_true', help='Perform face swap in both directions')
    
    args = parser.parse_args()
    
    try:
        # Initialize face analysis and swapper
        print("Initializing InsightFace models...")
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Try to load the swapper model with fallback options
        try:
            swapper = get_model('inswapper_128.onnx', download=True, download_zip=True)
        except Exception as e:
            print(f"Failed to download model automatically: {e}")
            print("\nPlease manually download the inswapper model:")
            print("1. Go to: https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx")
            print("2. Download the file")
            print("3. Place it in: C:\\Users\\omdee\\.insightface\\models\\inswapper_128.onnx")
            print("4. Run the script again")
            
            # Try to load if already exists
            model_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
            if os.path.exists(model_path):
                print(f"Found existing model at: {model_path}")
                swapper = get_model('inswapper_128.onnx', download=False)
            else:
                raise Exception("inswapper_128.onnx model not found and download failed")
        print("Models loaded successfully!")
        
        # Create output folder
        output_folder = create_output_folder(args.output_folder)
        
        # Load images
        print(f"Loading source image: {args.source}")
        source_img = load_image(args.source)
        
        print(f"Loading target image: {args.target}")
        target_img = load_image(args.target)
        
        # Get filenames for output naming
        source_name = get_filename_without_extension(args.source)
        target_name = get_filename_without_extension(args.target)
        
        # Check faces in both images
        source_faces = app.get(source_img)
        target_faces = app.get(target_img)
        
        print(f"Source image faces detected: {len(source_faces)}")
        print(f"Target image faces detected: {len(target_faces)}")
        
        if len(source_faces) == 0 or len(target_faces) == 0:
            print("Error: No faces detected in one or both images!")
            return
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Perform face swap: source face -> target image
        print("Performing face swap: source face -> target image...")
        result1 = swap_faces(swapper, app, source_img, target_img, 
                           args.source_face_idx, args.target_face_idx)
        
        output1_path = os.path.join(output_folder, f"{source_name}_to_{target_name}_{timestamp}.jpg")
        save_result(result1, output1_path)
        
        if args.both_ways:
            # Perform face swap: target face -> source image
            print("Performing face swap: target face -> source image...")
            result2 = swap_faces(swapper, app, target_img, source_img, 
                               args.target_face_idx, args.source_face_idx)
            
            output2_path = os.path.join(output_folder, f"{target_name}_to_{source_name}_{timestamp}.jpg")
            save_result(result2, output2_path)
            
            print(f"\nBoth face swaps completed!")
            print(f"Results saved in '{output_folder}' folder:")
            print(f"  - {os.path.basename(output1_path)}")
            print(f"  - {os.path.basename(output2_path)}")
        else:
            print(f"\nFace swap completed!")
            print(f"Result saved: {output1_path}")
            print(f"Tip: Use --both_ways or -b flag to generate both swap directions")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# Example usage:
# python deepfake.py --source image1.jpg --target image2.jpg
# python deepfake.py -s image1.jpg -t image2.jpg --both_ways
# python deepfake.py -s image1.jpg -t image2.jpg -o my_results --both_ways
