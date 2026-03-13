
import importlib

requirements = [
    'tweepy', 'python-dotenv', 'pandas', 'pyyaml', 'nltk', 
    'sklearn', 'joblib', 'matplotlib', 'seaborn', 'transformers', 
    'torch', 'datasets', 'evaluate', 'accelerate', 'scipy', 'tqdm'
]

missing = []
for req in requirements:
    try:
        if req == 'sklearn':
            importlib.import_module('sklearn')
        elif req == 'python-dotenv':
            importlib.import_module('dotenv')
        elif req == 'pyyaml':
            importlib.import_module('yaml')
        else:
            importlib.import_module(req)
    except ImportError:
        missing.append(req)

if missing:
    print(f"Missing modules: {', '.join(missing)}")
else:
    print("All modules are installed.")
