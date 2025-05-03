import cv2
import numpy as np
import dlib
import os
from collections import OrderedDict

# Define facial landmarks indices
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def rect_to_bb(rect):
    """Convert dlib rectangle to OpenCV format (x, y, w, h)"""
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    """Convert dlib shape to numpy array"""
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_face_landmarks(img, detector, predictor):
    """Detect face and extract landmarks"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    if len(rects) > 0:
        # Get largest face based on area
        areas = [(rect.right() - rect.left()) * (rect.bottom() - rect.top()) for rect in rects]
        largest_idx = np.argmax(areas)
        rect = rects[largest_idx]
        
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        return shape, rect
    return None, None

def extract_face_region(img, landmarks):
    """Extract convex hull of face based on landmarks"""
    hull_points = cv2.convexHull(landmarks)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_points, 255)
    
    # Apply mask to extract face region
    face_region = cv2.bitwise_and(img, img, mask=mask)
    return face_region, mask, hull_points

def get_affine_transform(source_landmarks, target_landmarks):
    """Calculate affine transform to align source face to target face"""
    # Select triangle of points (eyes and nose) for transformation
    source_triangle = np.float32([
        source_landmarks[36],  # Right eye corner
        source_landmarks[45],  # Left eye corner
        source_landmarks[33],  # Nose tip
    ])
    
    target_triangle = np.float32([
        target_landmarks[36],  # Right eye corner
        target_landmarks[45],  # Left eye corner
        target_landmarks[33],  # Nose tip
    ])
    
    # Get affine transform
    transform_matrix = cv2.getAffineTransform(source_triangle, target_triangle)
    return transform_matrix

def warp_and_blend_face(source_img, source_landmarks, target_img, target_landmarks):
    """Warp source face to match target face pose and blend them together"""
    # Get dimensions for warping
    h, w = target_img.shape[:2]
    
    # Get affine transform
    transform_matrix = get_affine_transform(source_landmarks, target_landmarks)
    
    # Warp source image
    warped_source = cv2.warpAffine(source_img, transform_matrix, (w, h))
    
    # Create target face mask
    target_hull_points = cv2.convexHull(target_landmarks)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, target_hull_points, 255)
    
    # Extract target face region for seamless cloning
    source_face_warped = cv2.bitwise_and(warped_source, warped_source, mask=mask)
    
    # Calculate center point for seamless cloning
    center = (int(np.mean(target_landmarks[:, 0])), int(np.mean(target_landmarks[:, 1])))
    
    # Apply seamless cloning
    output = cv2.seamlessClone(source_face_warped, target_img, mask, center, cv2.NORMAL_CLONE)
    
    return output

def create_face_model(source_image_paths):
    """Build a reference model for a person from multiple source images"""
    print("Creating face model from source images...")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    source_faces = []
    source_landmarks_list = []
    
    for img_path in source_image_paths:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            landmarks, rect = get_face_landmarks(img, detector, predictor)
            if landmarks is not None:
                face_region, mask, _ = extract_face_region(img, landmarks)
                source_faces.append(face_region)
                source_landmarks_list.append(landmarks)
                print(f"Processed: {img_path}")
            else:
                print(f"No face detected in: {img_path}")
        else:
            print(f"File not found: {img_path}")
    
    if not source_faces:
        print("No valid source faces found!")
        return None, None
        
    print(f"Successfully processed {len(source_faces)} source images")
    return source_faces, source_landmarks_list

def run_face_swap(source_image_paths):
    """Main function to run the face swap application"""
    # Check if we have the face landmark predictor file
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("Landmark predictor file not found!")
        print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract it and place it in the same directory as this script.")
        return
    
    # Initialize face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Build source face model
    source_faces, source_landmarks_list = create_face_model(source_image_paths)
    if source_faces is None:
        return
    
    # Select the first source face as default (you can implement more sophisticated selection)
    default_source_face = source_faces[0]
    default_source_landmarks = source_landmarks_list[0]
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open webcam!")
        return
    
    print("Starting face swap. Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Detect face landmarks in the webcam frame
        webcam_landmarks, webcam_rect = get_face_landmarks(frame, detector, predictor)
        
        if webcam_landmarks is not None:
            # Draw landmarks on display frame (optional)
            for (x, y) in webcam_landmarks:
                cv2.circle(display_frame, (x, y), 1, (0, 255, 0), -1)
            
            # Perform face swap
            try:
                result = warp_and_blend_face(default_source_face, default_source_landmarks, 
                                            frame, webcam_landmarks)
                cv2.imshow("Face Swap", result)
            except Exception as e:
                print(f"Error during face swap: {e}")
                cv2.imshow("Face Swap", display_frame)
        else:
            # If no face detected, show the original frame
            cv2.putText(display_frame, "No face detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Face Swap", display_frame)
        
        # Check for exit keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Face swap terminated")

if __name__ == "__main__":
    # Define source image paths here
    source_image_paths = [
        "source_face1.jpg",
        "source_face2.jpg",
        "source_face3.jpg"
    ]
    
    run_face_swap(source_image_paths)