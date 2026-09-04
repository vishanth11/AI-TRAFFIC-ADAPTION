import os

for root, dirs, files in os.walk("data/raw"):
    print("\nFolder:", root)

    for file in files:
        print("   ", file)