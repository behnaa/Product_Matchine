
import os
import csv
import shutil
import certifi
import pymongo
import pandas as pd
from PIL import Image
from pymongo.mongo_client import MongoClient

# ====== CONFIG ======
CA = certifi.where()
URI = ""
DB_NAME = "productDb"
COLLECTION_NAME = "products_direct"
ORIGINAL_CSV = r"E:\Assignment\data.csv"
ORIGINAL_IMAGE_FOLDER = r"E:\Assignment\data"
FILTERED_CSV = r"E:\Assignment\filtered_data.csv"
FILTERED_IMAGE_FOLDER = r"E:\Assignment\filtered_images"
VALID_CATEGORIES = [
    "Sports Shoes", "Casual Shoes", "Sandals", "FlipFlops", "Heels",
    "Flats", "Sneakers", "Boots", "Slippers", "Loafers", "Ballerinas"
]
RESIZE_DIM = (224, 224)

# ====== MONGODB CONNECTION ======
client = MongoClient(URI, tlsCAFile=CA)
db = client[DB_NAME]
products = db[COLLECTION_NAME]

# Clear the collection if rerunning
products.delete_many({})

# ====== SETUP FOLDER ======
if os.path.exists(FILTERED_IMAGE_FOLDER):
    shutil.rmtree(FILTERED_IMAGE_FOLDER)
os.makedirs(FILTERED_IMAGE_FOLDER, exist_ok=True)

# ====== LOAD & FILTER CSV ======
df = pd.read_csv(ORIGINAL_CSV)
filtered_df = df[df['category'].isin(VALID_CATEGORIES)].copy()
filtered_df.to_csv(FILTERED_CSV, index=False)

# Print category counts
category_counts = filtered_df['category'].value_counts()
print("\n Category-wise counts:")
print(category_counts)

# ====== PROCESS IMAGES & SAVE ======
for idx, row in filtered_df.iterrows():
    image_name = row['image']
    category = row['category']
    image_path = os.path.join(ORIGINAL_IMAGE_FOLDER, image_name)

    if os.path.exists(image_path):
        try:
            # Resize image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(RESIZE_DIM)

                # Save resized image in folder
                save_path = os.path.join(FILTERED_IMAGE_FOLDER, image_name)
                img.save(save_path)

                # Convert resized image to binary
                with open(save_path, "rb") as f:
                    binary_data = f.read()

                # Store in MongoDB (binary inside document)
                product_doc = {
                    "image_name": image_name,
                    "description": row.get("description", ""),
                    "display_name": row.get("display name", ""),
                    "category": category,
                    "image_data": binary_data  # Binary in document
                }
                products.insert_one(product_doc)
                print(f"[✔] Inserted {image_name} in category '{category}'")

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    else:
        print(f"Image not found: {image_name}")

print("\n Completed successfully.")
















# import sys
# import certifi
# import gridfs 
# import pymongo
# import os
# import csv
# import pandas as pd
# from pymongo.mongo_client import MongoClient
# ca = certifi.where()
# uri = "mongodb+srv://hafsabatool:333hafsa@cluster0.lmhxlia.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# client = MongoClient(uri, tlsCAFile=ca)
# db=client["productDb"]
# fs = gridfs.GridFS(db)
# products = db["products"]
# csv_file_path =r"E:\Assignment\data.csv"
# image_folder =r"E:\Assignment\data"
# # Read CSV using pandas
# df = pd.read_csv(csv_file_path)

# # Iterate through rows in DataFrame
# for _, row in df.iterrows():
#     image_name = row['image']
#     image_path = os.path.join(image_folder, image_name)

#     if os.path.exists(image_path):
#         with open(image_path, 'rb') as f:
#             image_data = f.read()
#             file_id = fs.put(image_data, filename=image_name)

#         # Create product document
#         product_doc = {
#             "image_name": image_name,
#             "description": row.get("description", ""),
#             "display_name": row.get("display name", ""),
#             "category": row.get("category", ""),
#             "image_file_id": file_id
#         }
#         print(f"Category: {product_doc['category']}, Image Name: {product_doc['image_name']}")
#         products.insert_one(product_doc)
#         # print(f"Inserted: {image_name}")
#     else:
#         print(f"Image file not found: {image_name}")
