import os
import shutil
import random

def split_dataset(source_dir, train_dir, test_dir, train_size):
    # Create train and test directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Get the list of class directories
    classes = os.listdir(source_dir)
    
    # Iterate over each class directory
    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir):
            # Create class directories in train and test directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)
            
            # Get list of images in the class directory
            images = os.listdir(class_dir)
            random.shuffle(images)
            
            # Split images into train and test sets
            train_images = images[:train_size]
            test_images = images[train_size:train_size+250]
            
            # Copy train images to train directory
            for image in train_images:
                src = os.path.join(class_dir, image)
                dst = os.path.join(train_class_dir, image)
                shutil.copy(src, dst)
            
            # Copy test images to test directory
            for image in test_images:
                src = os.path.join(class_dir, image)
                dst = os.path.join(test_class_dir, image)
                shutil.copy(src, dst)

# Paths to source, train, and test directories
source_dir = "og_images"  
train_dir = "dataset/train"
test_dir = "dataset/test"

# Split dataset with 750 images in train and 250 in test directories
split_dataset(source_dir, train_dir, test_dir, train_size=750)
