from sklearn.model_selection import train_test_split
import os

# Define directory and test size
dir_path = r"/vol/research/RobotFarming/Projects/hyper_transformer/datasets/enmap_gdal/n550_l12"
test_size = 0.2  # 20% for testing, 80% for training

# List all files in directory
all_files = os.listdir(dir_path)

# Split the files into training and test sets
train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42)

# Save the filenames of the training and test sets in separate text files
with open(os.path.join(dir_path, "train.txt"), "w") as f:
    f.write("\n".join(train_files))

with open(os.path.join(dir_path, "val.txt"), "w") as f:
    f.write("\n".join(test_files))

print('Done!')
