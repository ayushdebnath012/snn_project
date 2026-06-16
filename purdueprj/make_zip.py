import os
import zipfile

def make_kaggle_zip(source_dir, output_filename):
    print(f"Creating {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            if '.git' in root or '__pycache__' in root or 'env' in root or 'venv' in root or '.ipynb_checkpoints' in root:
                continue
            for file in files:
                if file.endswith('.zip'):
                    continue
                abs_path = os.path.join(root, file)
                # Relpath and replace slashes
                rel_path = os.path.relpath(abs_path, source_dir)
                archive_name = rel_path.replace('\\', '/')
                print(f"  Adding: {archive_name}")
                zipf.write(abs_path, archive_name)
    print("Done! Safe for Kaggle upload.")

if __name__ == '__main__':
    make_kaggle_zip('.', 'purdueprj_kaggle.zip')
