"""Quick test to verify patchify and patchify_mask directories"""
from pathlib import Path

# Check directories
image_dir = Path(__file__).parent
mask_dir = Path(__file__).parent.parent / "patchify_mask"

print("="*60)
print("DATA VERIFICATION")
print("="*60)

print(f"\nImage directory: {image_dir}")
print(f"  Exists: {image_dir.exists()}")
print(f"  PNG files: {len(list(image_dir.glob('*.png')))}")

print(f"\nMask directory: {mask_dir}")
print(f"  Exists: {mask_dir.exists()}")
print(f"  PNG files: {len(list(mask_dir.glob('*.png')))}")

# Check if they match
if image_dir.exists() and mask_dir.exists():
    images = sorted([f.name for f in image_dir.glob('*.png')])
    masks = sorted([f.name for f in mask_dir.glob('*.png')])
    
    matching = set(images) & set(masks)
    print(f"\n Matching pairs: {len(matching)}")
    
    if len(matching) > 0:
        print(f"\n✓ Ready to train! Found {len(matching)} image-mask pairs")
        print(f"\nFirst 5 pairs:")
        for name in list(matching)[:5]:
            print(f"  - {name}")
    else:
        print("\n✗ No matching image-mask pairs found!")
else:
    print("\n✗ One or both directories not found!")
