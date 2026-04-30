import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
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
    lines_raw = cv2.HoughLinesP(edges, 1, np.pi/180, 60, 70, 15)
    if lines_raw is None:
        return []
    
    # Split lines by slope sign for 2-point perspective
    positive_slope = []
    negative_slope = []
    
    for l in lines_raw:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1) + 1e-6))
        if 10 < abs(angle) < 80:
            if angle > 0:
                positive_slope.append((x1, y1, x2, y2))
            else:
                negative_slope.append((x1, y1, x2, y2))
    
    return positive_slope, negative_slope

def line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

def find_vp_from_lines(lines, img_shape):
    """Find vanishing point from a set of lines"""
    h, w = img_shape[:2]
    if len(lines) < 3:
        return None
    
    intersections = []
    n = len(lines)
    indices = list(range(n))
    if n > 80:
        np.random.seed(42)
        indices = np.random.choice(n, 80, replace=False)
    
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            pt = line_intersection(lines[indices[i]], lines[indices[j]])
            if pt:
                x, y = pt
                if -3*w < x < 3*w and -3*h < y < 3*h:
                    intersections.append(pt)
    
    if len(intersections) < 3:
        return None
    
    pts = np.array(intersections)
    
    # Use DBSCAN for clustering
    diagonal = np.sqrt(h**2 + w**2)
    pts_norm = pts / diagonal
    
    db = DBSCAN(eps=0.2, min_samples=3).fit(pts_norm)
    
    # Get largest cluster
    labels = db.labels_
    if len(set(labels) - {-1}) == 0:
        return None
    
    largest_cluster = max(set(labels) - {-1}, key=lambda l: np.sum(labels == l))
    mask = labels == largest_cluster
    cluster_pts = pts[mask]
    
    vp = np.median(cluster_pts, axis=0)
    return {"point": vp, "num_intersections": len(cluster_pts)}

def compute_horizon_line(vp1, vp2, img_shape):
    """Compute line through two vanishing points"""
    h, w = img_shape[:2]
    x1, y1 = vp1
    x2, y2 = vp2
    
    if abs(x2 - x1) < 1e-6:
        return [(x1, 0), (x1, h)]
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    # Extend to image boundaries
    points = []
    
    y_left = intercept
    if 0 <= y_left <= h:
        points.append((0, y_left))
    
    y_right = slope * w + intercept
    if 0 <= y_right <= h:
        points.append((w, y_right))
    
    if len(points) >= 2:
        return points[:2]
    
    return [(0, intercept), (w, slope * w + intercept)]

def visualize_2point(img, pos_lines, neg_lines, vp1, vp2, horizon_pts, output_path=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img_rgb)
    
    # Draw lines with different colors
    for x1, y1, x2, y2 in pos_lines:
        ax.plot([x1, x2], [y1, y2], color='cyan', linewidth=0.8, alpha=0.4)
    for x1, y1, x2, y2 in neg_lines:
        ax.plot([x1, x2], [y1, y2], color='lime', linewidth=0.8, alpha=0.4)
    
    # Draw vanishing points
    ax.plot(vp1[0], vp1[1], 'X', color='red', markersize=14, 
           markeredgecolor='white', markeredgewidth=2, label='VP1 (Left)')
    ax.plot(vp2[0], vp2[1], 'X', color='orange', markersize=14, 
           markeredgecolor='white', markeredgewidth=2, label='VP2 (Right)')
    
    ax.annotate(f'VP1: ({int(vp1[0])}, {int(vp1[1])})', 
               xy=(vp1[0], vp1[1]), xytext=(10, 10), 
               textcoords='offset points', color='red', fontweight='bold')
    ax.annotate(f'VP2: ({int(vp2[0])}, {int(vp2[1])})', 
               xy=(vp2[0], vp2[1]), xytext=(10, 10), 
               textcoords='offset points', color='orange', fontweight='bold')
    
    # Draw Vanishing Line & Horizon Line
    if horizon_pts:
        ax.plot([horizon_pts[0][0], horizon_pts[1][0]], 
               [horizon_pts[0][1], horizon_pts[1][1]], 
               color='magenta', linewidth=3, linestyle='--', 
               label='Vanishing Line = Horizon Line')
    
    plt.title('Single View Geometry: 2-POINT PERSPECTIVE\nTwo Vanishing Points + Horizon Line', 
             fontsize=14, fontweight='bold')
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"\n💾 Plot saved to: {output_path}")
    plt.show()

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"')
    
    if not os.path.isfile(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return
    
    print("\n" + "="*60)
    print("📐 2-POINT PERSPECTIVE ANALYZER")
    print("="*60)
    
    img = load_image(image_path)
    edges = preprocess(img)
    pos_lines, neg_lines = detect_lines(edges)
    
    print(f"✓ Detected {len(pos_lines)} lines (positive slope)")
    print(f"✓ Detected {len(neg_lines)} lines (negative slope)")
    
    vp1 = find_vp_from_lines(pos_lines, img.shape)
    vp2 = find_vp_from_lines(neg_lines, img.shape)
    
    if vp1 and vp2:
        vp1_pt = vp1["point"]
        vp2_pt = vp2["point"]
        
        print(f"\n📍 Vanishing Point 1 (Left): ({int(vp1_pt[0])}, {int(vp1_pt[1])})")
        print(f"   Based on: {vp1['num_intersections']} intersections")
        print(f"\n📍 Vanishing Point 2 (Right): ({int(vp2_pt[0])}, {int(vp2_pt[1])})")
        print(f"   Based on: {vp2['num_intersections']} intersections")
        
        horizon_pts = compute_horizon_line(vp1_pt, vp2_pt, img.shape)
        
        print(f"\n📐 Horizon Line: passes through both vanishing points")
        print("\n✅ Analysis complete - 2-Point Perspective confirmed")
        
        visualize_2point(img, pos_lines, neg_lines, vp1_pt, vp2_pt, horizon_pts, "2point_result.png")
    else:
        print("\n❌ Could not find both vanishing points.")
        print("   Make sure your image has 2-point perspective (e.g., building corner)")

if __name__ == "__main__":
    main()