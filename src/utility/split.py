import shutil
from pathlib import Path

def split_dataset(source_dir="DataSet/Testing"):
    """Permanently splits Testing folder into Validation/Test (50/50)"""
    src = Path(source_dir)
    val = src.parent/"Validation"
    test = src.parent/"Test"

    # Create target folders
    val.mkdir(exist_ok=True)
    test.mkdir(exist_ok=True)
    
    for class_dir in src.iterdir():
        if not class_dir.is_dir():
            continue
            
        # Create class subfolders
        (val/class_dir.name).mkdir()
        (test/class_dir.name).mkdir()
        
        # Get sorted files and split point
        files = sorted(f for f in class_dir.iterdir() if f.is_file())
        mid = len(files)//2
        
        # Move files
        for i, f in enumerate(files):
            dest = val if i < mid else test
            shutil.move(str(f), str(dest/class_dir.name/f.name))


if __name__ == "__main__":
    split_dataset()