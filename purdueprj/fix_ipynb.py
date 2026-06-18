import json
import re

with open("run_kaggle.ipynb", "r", encoding="utf-8") as f:
    text = f.read()

# Let's find the unescaped literal string in "source": [ ... ]
# We know it starts with "if not os.path.exists('run_all_experiments.py'):\nimport shutil

start_str = "\"if not os.path.exists('run_all_experiments.py'):"
idx = text.find(start_str)

if idx != -1:
    # Find the end of this string by looking for the trailing closing quote before the \n"
    # Actually, the block ends with: ...purdueprj-code.')\n",
    end_str = "purdueprj-code.')\\n\","
    end_idx = text.find(end_str, idx)
    
    if end_idx != -1:
        end_idx += len(end_str)
        
        # Replace this whole block with the correct JSON array of strings
        correct_lines = [
            "if not os.path.exists('run_all_experiments.py'):\\n",
            "    import shutil\\n",
            "    purdue_code_dir = '/kaggle/input/datasets/ayushdebnath0123/purdueprj-code'\\n",
            "    if os.path.exists(purdue_code_dir):\\n",
            "        print(f'\\\\n⏳ Found uploaded codebase at {purdue_code_dir}')\\n",
            "        print('\\\\n⏳ Copying files to /kaggle/working/...')\\n",
            "        for item in os.listdir(purdue_code_dir):\\n",
            "            s = os.path.join(purdue_code_dir, item)\\n",
            "            d = os.path.join(LOCAL_PROJECT, item)\\n",
            "            if os.path.isdir(s):\\n",
            "                if not os.path.exists(d):\\n",
            "                    shutil.copytree(s, d)\\n",
            "            else:\\n",
            "                if not os.path.exists(d):\\n",
            "                    shutil.copy2(s, d)\\n",
            "        print('\\\\n✅ Copy complete!\\\\n')\\n",
            "    else:\\n",
            "        print('\\\\n⚠️ WARNING: Could not find purdueprj dataset! Make sure you added it exactly as Ayushdebnath0123/purdueprj-code.')\\n"
        ]
        
        replacement = ",\n    ".join(f'"{line}"' for line in correct_lines) + ","
        
        new_text = text[:idx] + replacement + text[end_idx:]
        
        with open("run_kaggle.ipynb", "w", encoding="utf-8") as f:
            f.write(new_text)
        print("Replaced bad block successfully.")
        
        # Verify JSON validity
        try:
            json.loads(new_text)
            print("JSON is now valid.")
        except json.JSONDecodeError as e:
            print("JSON still invalid:", e)
    else:
        print("Could not find end of broken block.")
else:
    print("Could not find start of broken block.")
