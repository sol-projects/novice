import os
import random
import shutil
from PIL import Image

def resize_and_save(src_path, dst_path, size=(50, 50)):
    """
    Open an image from src_path, resize it to 50×50, and save to dst_path.
    """
    with Image.open(src_path) as img:
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(dst_path, format="JPEG")  # always save as .jpg

def prepare_data():
    source_root = "data/raw"
    target_root = "data"
    
    cars_src_dir = os.path.join(source_root, "cars")
    cars_dst_dir = os.path.join(target_root, "cars")
    os.makedirs(cars_dst_dir, exist_ok=True)
    
    car_files = [
        f for f in os.listdir(cars_src_dir) 
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(car_files)
    car_files = car_files[:1500]  # pick up to 1500
    
    for i, filename in enumerate(car_files):
        src_path = os.path.join(cars_src_dir, filename)
        new_name = f"{i}.jpg"
        dst_path = os.path.join(cars_dst_dir, new_name)
        resize_and_save(src_path, dst_path, size=(50, 50))
    print(f"Copied and resized {len(car_files)} car images to: {cars_dst_dir}")
    
    cats_src_dir = os.path.join(source_root, "cats")
    cats_dst_dir = os.path.join(target_root, "cats")
    os.makedirs(cats_dst_dir, exist_ok=True)
    
    cat_files = [
        f for f in os.listdir(cats_src_dir) 
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    for i, filename in enumerate(cat_files):
        src_path = os.path.join(cats_src_dir, filename)
        new_name = f"{i}.jpg"
        dst_path = os.path.join(cats_dst_dir, new_name)
        resize_and_save(src_path, dst_path, size=(50, 50))
    print(f"Copied and resized {len(cat_files)} cat images to: {cats_dst_dir}")
    
    dogs_src_dir = os.path.join(source_root, "dogs")
    dogs_dst_dir = os.path.join(target_root, "dogs")
    os.makedirs(dogs_dst_dir, exist_ok=True)
    
    dog_files = [
        f for f in os.listdir(dogs_src_dir) 
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    for i, filename in enumerate(dog_files):
        src_path = os.path.join(dogs_src_dir, filename)
        new_name = f"{i}.jpg"
        dst_path = os.path.join(dogs_dst_dir, new_name)
        resize_and_save(src_path, dst_path, size=(50, 50))
    print(f"Copied and resized {len(dog_files)} dog images to: {dogs_dst_dir}")
    
    # TODO Handle 'emergency_vehicle' different raw format
    
    ev_src_dir = os.path.join(source_root, "emergency_vehicle")
    ev_dst_dir = os.path.join(target_root, "emergency_vehicle")
    os.makedirs(ev_dst_dir, exist_ok=True)
    
    ev_files = [
        f for f in os.listdir(ev_src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    for i, filename in enumerate(ev_files):
        src_path = os.path.join(ev_src_dir, filename)
        new_name = f"{i}.jpg"
        dst_path = os.path.join(ev_dst_dir, new_name)
        resize_and_save(src_path, dst_path, size=(50, 50))
    print(f"Copied and resized {len(ev_files)} emergency_vehicle images to: {ev_dst_dir}")
    
    csv_src = os.path.join(source_root, "train.csv")
    csv_dst = os.path.join(ev_dst_dir, "train.csv")
    if os.path.exists(csv_src):
        shutil.copy(csv_src, csv_dst)
        print(f"Copied train.csv to: {csv_dst}")
    else:
        print("train.csv not found in raw directory, skipping copy.")

if __name__ == "__main__":
    prepare_data()
    print("Data preparation complete.")

