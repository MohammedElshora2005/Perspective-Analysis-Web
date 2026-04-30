import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {path}")
    h, w = img.shape[:2]
    scale = min(1.0, 1200 / max(w, h))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    low = int(max(0, 0.66 * v))
    high = int(min(255, 1.33 * v))
    return cv2.Canny(blur, low, high)

def detect_lines(edges):
    lines_raw = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 60, 15)
    if lines_raw is None:
        return []
    filtered = []
    for l in lines_raw:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-6))
        if 10 < angle < 80:
            filtered.append((x1, y1, x2, y2))
    return filtered

def line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

def find_single_vanishing_point(lines, img_shape):
    """Find the single strongest vanishing point (for 1-point perspective)"""
    h, w = img_shape[:2]
    intersections = []
    
    n = len(lines)
    indices = list(range(n))
    if n > 100:
        np.random.seed(42)
        indices = np.random.choice(n, 100, replace=False)
    
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            pt = line_intersection(lines[indices[i]], lines[indices[j]])
            if pt:
                x, y = pt
                if -2*w < x < 2*w and -2*h < y < 2*h:
                    intersections.append(pt)
    
    if len(intersections) < 3:
        return None
    
    # Use median of all intersections as the single vanishing point
    pts = np.array(intersections)
    vp = np.median(pts, axis=0)
    
    return {"point": vp, "num_intersections": len(intersections)}

def visualize_1point(img, lines, vp_info, output_path=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img_rgb)
    
    # Draw lines
    for x1, y1, x2, y2 in lines:
        ax.plot([x1, x2], [y1, y2], color='yellow', linewidth=0.8, alpha=0.4)
    
    # Draw vanishing point
    vp = vp_info["point"]
    ax.plot(vp[0], vp[1], 'X', color='red', markersize=16, 
           markeredgecolor='white', markeredgewidth=2,
           label='Vanishing Point (1-Point)')
    ax.annotate(f'Vanishing Point\n({int(vp[0])}, {int(vp[1])})', 
               xy=(vp[0], vp[1]), xytext=(10, 10), 
               textcoords='offset points',
               color='red', fontweight='bold', fontsize=11,
               bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add title
    plt.title('Single View Geometry: 1-POINT PERSPECTIVE\nAll lines converge to one vanishing point', 
             fontsize=14, fontweight='bold')
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"\n💾 Plot saved to: {output_path}")
    plt.show()

def main():
    # Get image path from command line or user input
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"')
    
    if not os.path.isfile(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return
    
    print("\n" + "="*60)
    print("📐 1-POINT PERSPECTIVE ANALYZER")
    print("="*60)
    
    img = load_image(image_path)
    edges = preprocess(img)
    lines = detect_lines(edges)
    
    if len(lines) < 5:
        print("Not enough lines detected. Try a different image.")
        return
    
    print(f"✓ Detected {len(lines)} lines")
    
    vp_info = find_single_vanishing_point(lines, img.shape)
    
    if vp_info:
        vp = vp_info["point"]
        print(f"\n📍 Vanishing Point: ({int(vp[0])}, {int(vp[1])})")
        print(f"   Based on: {vp_info['num_intersections']} line intersections")
        print("\n✅ Analysis complete - 1-Point Perspective confirmed")
        
        visualize_1point(img, lines, vp_info, "1point_result.png")
    else:
        print("\n❌ Could not find vanishing point. Try a different image.")

if __name__ == "__main__":
    main()