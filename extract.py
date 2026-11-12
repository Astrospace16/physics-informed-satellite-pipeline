import os
import re

MD_FILE = "complete_repository_codebase (1).md" # Make sure this matches your exact filename

with open(MD_FILE, "r", encoding="utf-8") as f:
    content = f.read()

pattern = re.compile(r"## File: `(.*?)`\n```[a-zA-Z]*\n(.*?)\n```", re.DOTALL)
matches = pattern.findall(content)

for filepath, code in matches:
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Created: {filepath}")

print(f"\nExtracted {len(matches)} files successfully!")