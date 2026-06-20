import os
import shutil
import glob
import random
from pathlib import Path

def main():
    src_img = r"C:\Users\enosh\oralguard\data\finetune\periapical_yolo\images"
    src_lbl = r"C:\Users\enosh\oralguard\data\finetune\periapical_yolo\labels"

    imgs = glob.glob(os.path.join(src_img, '*.jpg'))
    print(f"Adding {len(imgs)} periapical images to combined dataset")

    random.seed(99)
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * 0.80)
    n_val = int(n * 0.15)

    splits = {
        'train': imgs[:n_train],
        'val': imgs[n_train:n_train+n_val],
        'test': imgs[n_train+n_val:]
    }

    added = 0
    for split_name, split_imgs in splits.items():
        img_out = f"C:\\Users\\enosh\\oralguard\\data\\combined\\images\\{split_name}"
        lbl_out = f"C:\\Users\\enosh\\oralguard\\data\\combined\\labels\\{split_name}"
        
        for img_src in split_imgs:
            base = Path(img_src).stem
            unique = f"peri_{added:06d}"
            
            shutil.copy2(
                img_src,
                os.path.join(img_out, unique + ".jpg")
            )
            
            lbl_src = os.path.join(src_lbl, base + ".txt")
            lbl_dst = os.path.join(lbl_out, unique + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, lbl_dst)
            else:
                open(lbl_dst, 'w').close()
            
            added += 1

    print(f"Added {added} periapical images to combined dataset")

    # Count class distribution
    all_labels = glob.glob(
        r"C:\Users\enosh\oralguard\data\combined\labels\train\*.txt"
    )
    counts = {0:0, 1:0, 2:0, 3:0}
    for lbl in all_labels:
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    if cls in counts:
                        counts[cls] += 1
    names = ['caries','deep_caries','periapical_lesion','impacted_tooth']
    total = sum(counts.values()) or 1
    for i, name in enumerate(names):
        print(f"{name}: {counts[i]} ({counts[i]/total*100:.1f}%)")

if __name__ == "__main__":
    main()
