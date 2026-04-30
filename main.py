import subprocess
import sys
import os

def main():
    print("\n" + "="*70)
    print("🎯 SINGLE VIEW GEOMETRY - MAIN CONTROLLER")
    print("="*70)
    
    # Step 1: Get image path
    image_path = input("\n📁 Enter image path: ").strip().strip('"')
    
    if not os.path.isfile(image_path):
        print(f"❌ Error: Image not found at '{image_path}'")
        return
    
    # Step 2: Ask user for perspective type
    print("\n" + "="*70)
    print("📷 SELECT PERSPECTIVE TYPE")
    print("="*70)
    print("\nLook at your image and choose:")
    print("   [1] 1-POINT perspective")
    print("       → Examples: corridor, road, tunnel, railway tracks")
    print("       → Lines converge to a SINGLE vanishing point")
    print()
    print("   [2] 2-POINT perspective")
    print("       → Examples: corner of a building, cube, room corner")
    print("       → Lines converge to TWO vanishing points (left and right)")
    print()
    print("   [3] 3-POINT perspective")
    print("       → Examples: tall building from below, skyscraper")
    print("       → Lines converge to THREE vanishing points")
    print("-"*50)
    
    while True:
        try:
            choice = int(input("\n👉 Enter your choice (1/2/3): "))
            if choice in [1, 2, 3]:
                break
            print("❌ Please enter 1, 2, or 3.")
        except ValueError:
            print("❌ Invalid input. Please enter 1, 2, or 3.")
    
    # Step 3: Run appropriate script
    scripts = {
        1: "vp_1point.py",
        2: "vp_2point.py",
        3: "vp_3point.py"
    }
    
    script_path = scripts[choice]
    
    if not os.path.isfile(script_path):
        print(f"\n❌ Error: {script_path} not found in current directory")
        print("Make sure all script files are in the same folder:")
        print("   - main.py")
        print("   - vp_1point.py")
        print("   - vp_2point.py")
        print("   - vp_3point.py")
        return
    
    print(f"\n🚀 Running {choice}-point perspective analyzer...")
    print("="*70)
    
    # Pass image path to the child script
    result = subprocess.run([sys.executable, script_path, image_path], 
                          capture_output=False, 
                          text=True)
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()