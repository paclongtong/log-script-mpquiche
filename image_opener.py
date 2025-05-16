from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# your base paths
base = Path("/home/paul/data_quiche/logs_backup/stack_pathfixed_100MB_symmetric")
client_dir = base / "client"
server_dir = base / "server"
output_path = Path("/home/paul/data_quiche/logs_backup/stack_pathfixed_100MB_symmetric/compare")
output_path.mkdir(parents=True, exist_ok=True)

# labels you want in your titles
label1 = "client"
label2 = "server"

# collect images
valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
imgs1 = sorted(f for f in client_dir.iterdir() if f.suffix.lower() in valid_exts)
imgs2 = sorted(f for f in server_dir.iterdir() if f.suffix.lower() in valid_exts)

# match by filename
pairs = []
for img1 in imgs1:
    img2 = server_dir / img1.name
    if img2.exists():
        pairs.append((img1, img2))
    else:
        print(f"No match for {img1.name}")

# display side-by-side with your labels
for img1, img2 in pairs:
    im1 = Image.open(img1)
    im2 = Image.open(img2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(im1)
    ax1.set_title(f"{img1.name} ({label1})")
    ax1.axis("off")
    
    ax2.imshow(im2)
    ax2.set_title(f"{img2.name} ({label2})")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.show()
    save_name = img1.stem + "comparison" + img1.suffix
    fig.savefig(output_path / save_name)
