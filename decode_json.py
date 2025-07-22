import sys
import json

data = sys.stdin.read()
parsed = json.loads(data)
print(json.dumps(parsed, indent=2, ensure_ascii=False))
