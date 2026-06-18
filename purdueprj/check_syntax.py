import json
nb = json.load(open("run_kaggle.ipynb", encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try:
            if not src.startswith("!"):
                compile(src, f"cell_{i}", "exec")
        except Exception as e:
            print(f"Error in cell {i}: {e}")
print("All cells OK")
