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
    # Don't resize too much to preserve details
    scale = min(1.0, 1000 / max(w, h))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Use adaptive threshold for better edge detection
    edges = cv2.Canny(blur, 30, 100)
    return edges

def detect_all_lines(edges):
    """Detect all lines without filtering by angle"""
    lines_raw = cv2.HoughLinesP(edges, 1, np.pi/180,
                                 50,  # threshold
                                 40,  # min line length
                                 10)  # max gap
    if lines_raw is None:
        return []
    
    lines = []
    for l in lines_raw:
        x1, y1, x2, y2 = l[0]
        # Calculate angle
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6))
        
        # Keep all non-horizontal lines (allow vertical for 3-point)
        if 15 < angle < 85:  # Lines between 15 and 85 degrees
            lines.append((x1, y1, x2, y2))
    
    return lines

def line_intersection(l1, l2):
    """Calculate intersection point of two lines"""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    
    # Line equations: A1*x + B1*y = C1
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    
    det = A1 * B2 - A2 * B1
    
    if abs(det) < 1e-6:
        return None
    
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    
    return (x, y)

def classify_lines_by_orientation(lines, img_shape):
    """
    Classify lines into three groups for 3-point perspective:
    - Group 1: Lines that converge to left VP (positive slope, right-leaning)
    - Group 2: Lines that converge to right VP (negative slope, left-leaning)  
    - Group 3: Vertical-ish lines (slope > 1 in magnitude)
    """
    h, w = img_shape[:2]
    
    left_group = []    # Positive slope, more horizontal
    right_group = []   # Negative slope, more horizontal
    vertical_group = [] # Steep lines (near vertical)
    
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) < 1e-6:
            slope_magnitude = float('inf')
        else:
            slope = dy / dx
            slope_magnitude = abs(slope)
        
        # Classify based on slope
        if slope_magnitude > 1.5:  # Steep lines (vertical-ish)
            vertical_group.append((x1, y1, x2, y2))
        else:  # More horizontal lines
            if dy / (dx + 1e-6) > 0:  # Positive slope
                left_group.append((x1, y1, x2, y2))
            else:  # Negative slope
                right_group.append((x1, y1, x2, y2))
    
    return left_group, right_group, vertical_group

def find_vanishing_point_from_group(lines, img_shape, search_scale=3.0):
    """
    Find vanishing point from a group of lines by finding their intersection cluster
    """
    h, w = img_shape[:2]
    
    if len(lines) < 4:
        return None
    
    # Calculate all pairwise intersections
    intersections = []
    n = len(lines)
    
    # Sample to avoid O(n²) explosion
    max_samples = min(n, 40)
    indices = list(range(n))
    if n > max_samples:
        np.random.seed(42)
        indices = np.random.choice(n, max_samples, replace=False)
    
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            pt = line_intersection(lines[indices[i]], lines[indices[j]])
            if pt:
                x, y = pt
                # Accept intersections within a wide range
                if -search_scale*w < x < search_scale*w and -search_scale*h < y < search_scale*h:
                    intersections.append(pt)
    
    if len(intersections) < 5:
        return None
    
    pts = np.array(intersections)
    
    # Cluster the intersections
    diagonal = np.sqrt(h**2 + w**2)
    pts_norm = pts / diagonal
    
    # Try different eps values
    best_vp = None
    best_size = 0
    
    for eps in [0.1, 0.15, 0.2, 0.25, 0.3]:
        db = DBSCAN(eps=eps, min_samples=3).fit(pts_norm)
        labels = db.labels_
        
        # Find largest cluster
        unique_labels = set(labels) - {-1}
        if len(unique_labels) == 0:
            continue
        
        for lbl in unique_labels:
            mask = labels == lbl
            cluster_size = np.sum(mask)
            if cluster_size > best_size:
                cluster_pts = pts[mask]
                best_vp = np.median(cluster_pts, axis=0)
                best_size = cluster_size
    
    if best_vp is None:
        # Fallback: use median of all intersections
        best_vp = np.median(pts, axis=0)
        best_size = len(pts)
    
    return {"point": best_vp, "size": best_size}

def compute_vanishing_lines(vps, img_shape):
    """Compute lines connecting each pair of vanishing points"""
    lines_info = []
    
    if len(vps) >= 2:
        lines_info.append(("VP1-VP2 (Horizon)", vps[0]["point"], vps[1]["point"], 'cyan', '--'))
    if len(vps) >= 3:
        lines_info.append(("VP1-VP3", vps[0]["point"], vps[2]["point"], 'magenta', '-.'))
        lines_info.append(("VP2-VP3", vps[1]["point"], vps[2]["point"], 'yellow', ':'))
    
    return lines_info

def draw_line_through_points(ax, p1, p2, img_shape, color, linestyle, label):
    """Draw line that passes through two points and extends across image"""
    h, w = img_shape[:2]
    x1, y1 = p1
    x2, y2 = p2
    
    if abs(x2 - x1) < 1e-6:
        # Vertical line
        ax.axvline(x=x1, color=color, linestyle=linestyle, linewidth=2, alpha=0.7, label=label)
        return
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    # Extend line far beyond image boundaries
    x_start = -w
    x_end = 2 * w
    y_start = slope * x_start + intercept
    y_end = slope * x_end + intercept
    
    ax.plot([x_start, x_end], [y_start, y_end], color=color, 
           linestyle=linestyle, linewidth=2, alpha=0.7, label=label)

def visualize_3point(img, lines, vps, output_path=None):
    """Visualize 3-point perspective with all components"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img_rgb)
    
    # Draw detected lines (very light)
    for x1, y1, x2, y2 in lines:
        ax.plot([x1, x2], [y1, y2], color='yellow', linewidth=0.5, alpha=0.2)
    
    # Colors and labels for VPs
    vp_colors = ['red', 'green', 'blue']
    vp_labels = ['Left Vanishing Point', 'Right Vanishing Point', 'Vertical Vanishing Point']
    
    # Draw vanishing points
    for i, vp_info in enumerate(vps[:3]):
        vp = vp_info["point"]
        color = vp_colors[i]
        
        ax.plot(vp[0], vp[1], 'X', color=color, markersize=18, 
               markeredgecolor='white', markeredgewidth=2, label=vp_labels[i])
        
        ax.annotate(f'{vp_labels[i]}\n({int(vp[0])}, {int(vp[1])})', 
                   xy=(vp[0], vp[1]), xytext=(10, -15 if i == 2 else 10), 
                   textcoords='offset points',
                   color=color, fontweight='bold', fontsize=10,
                   bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Draw vanishing lines
    if len(vps) >= 2:
        vp1 = vps[0]["point"]
        vp2 = vps[1]["point"]
        draw_line_through_points(ax, vp1, vp2, img.shape, 'cyan', '--', 'Vanishing Line (Horizon)')
    
    if len(vps) >= 3:
        vp1 = vps[0]["point"]
        vp3 = vps[2]["point"]
        vp2 = vps[1]["point"]
        draw_line_through_points(ax, vp1, vp3, img.shape, 'magenta', '-.', 'Vanishing Line (Left-Vertical)')
        draw_line_through_points(ax, vp2, vp3, img.shape, 'orange', ':', 'Vanishing Line (Right-Vertical)')
    
    # Title
    if len(vps) >= 3:
        title = '3-POINT PERSPECTIVE ANALYSIS\nThree Vanishing Points with Connecting Lines'
    elif len(vps) == 2:
        title = 'PERSPECTIVE ANALYSIS (2 of 3 Vanishing Points Found)'
    else:
        title = f'PERSPECTIVE ANALYSIS ({len(vps)} Vanishing Point(s) Found)'
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Adjust limits to show all VPs
    all_x = [vp["point"][0] for vp in vps[:3]] + [0, img.shape[1]]
    all_y = [vp["point"][1] for vp in vps[:3]] + [0, img.shape[0]]
    
    margin = max(300, (max(all_x) - min(all_x)) * 0.3)
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(max(all_y) + margin, min(all_y) - margin)
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle=':')
    
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
    print("🎯 3-POINT PERSPECTIVE ANALYZER")
    print("="*60)
    
    # Load and process
    img = load_image(image_path)
    h, w = img.shape[:2]
    print(f"Image size: {w} x {h}")
    
    edges = preprocess(img)
    all_lines = detect_all_lines(edges)
    
    print(f"✓ Detected {len(all_lines)} lines")
    
    if len(all_lines) < 15:
        print("⚠️ Not enough lines detected. Try a different image with more straight lines.")
        return
    
    # Classify lines into three groups
    left_group, right_group, vertical_group = classify_lines_by_orientation(all_lines, img.shape)
    
    print(f"\n📊 Line classification:")
    print(f"   Left-converging lines: {len(left_group)}")
    print(f"   Right-converging lines: {len(right_group)}")
    print(f"   Vertical lines: {len(vertical_group)}")
    
    # Find vanishing points from each group
    vps = []
    
    # Try to find 3 VPs
    vp_left = find_vanishing_point_from_group(left_group, img.shape, search_scale=4.0)
    vp_right = find_vanishing_point_from_group(right_group, img.shape, search_scale=4.0)
    vp_vertical = find_vanishing_point_from_group(vertical_group, img.shape, search_scale=3.0)
    
    if vp_left:
        vps.append(vp_left)
        print(f"\n📍 Left VP: ({int(vp_left['point'][0])}, {int(vp_left['point'][1])})")
        print(f"   Based on {vp_left['size']} intersections")
    
    if vp_right:
        vps.append(vp_right)
        print(f"\n📍 Right VP: ({int(vp_right['point'][0])}, {int(vp_right['point'][1])})")
        print(f"   Based on {vp_right['size']} intersections")
    
    if vp_vertical:
        vps.append(vp_vertical)
        print(f"\n📍 Vertical VP: ({int(vp_vertical['point'][0])}, {int(vp_vertical['point'][1])})")
        print(f"   Based on {vp_vertical['size']} intersections")
    
    # Sort VPs by x-coordinate (left to right)
    vps.sort(key=lambda v: v['point'][0])
    
    print("\n" + "-"*40)
    
    if len(vps) >= 3:
        print("\n✅ SUCCESS: Found 3 Vanishing Points!")
        print("   → This confirms 3-Point Perspective")
        print("   → Camera is tilted (looking up/down at a tall structure)")
    elif len(vps) == 2:
        print("\n⚠️ Found 2 Vanishing Points (may be 2-point perspective)")
        print("   → Try an image of a tall building from below")
    else:
        print("\n❌ Found insufficient vanishing points")
        print("   → For 3-point perspective, use images of:")
        print("      - Tall buildings photographed from ground level")
        print("      - Skyscrapers looking up")
        print("      - Cityscapes from a high angle")
    
    # Visualize
    if vps:
        visualize_3point(img, all_lines, vps, "3point_result.png")
    else:
        print("\n❌ No vanishing points to display")

if __name__ == "__main__":
    main()