
# CiRA batch loader: returns one shuffled image and category per loop iteration.
import os
import random

# Define batch paths, mode, and reproducible shuffle seed.
BATCH_ROOT = r"C:\cira_batch_test"
INDEX_FILE = r"C:\cira_batch_test\batch_index.txt"
STOP_FILE = r"C:\cira_batch_test\stop.txt"
MODE = "monitor"
SHUFFLE_SEED = 2030

# Only these category folders are included.
VALID_CATEGORIES = [
	"bottle", "cable", "capsule", "carpet", "grid",
	"hazelnut", "leather", "metal_nut", "pill", "screw",
	"tile", "toothbrush", "transistor", "wood", "zipper"
]

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ============================================================
# 1. Stop check
# If stop.txt exists, stop the loop and delete stop.txt automatically.
# ============================================================

# Stop the batch when stop.txt is present.
if_stop = os.path.exists(STOP_FILE)

if if_stop:
	try:
		os.remove(STOP_FILE)
		stop_deleted = True
	except Exception as e:
		stop_deleted = False

	out = {
		"status": "STOPPED",
		"have_img": False,
		"message": "Batch stopped by user. stop.txt detected and deleted.",
		"stop_file_deleted": stop_deleted,
		"result_text": "Batch stopped by user.",
		"led_status": 0,
		"image_path": "",
		"category": "",
		"mode": "stopped",
		"index": 0,
		"total": 0
	}

else:
	image_list = []

	# ========================================================
	# 2. Check batch folder
	# ========================================================

	if not os.path.exists(BATCH_ROOT):
		out = {
			"status": "ERROR",
			"have_img": False,
			"message": "Batch root folder does not exist.",
			"batch_root": BATCH_ROOT,
			"result_text": "Batch root folder does not exist.",
			"led_status": 0,
			"image_path": ""
		}

	else:
		# ====================================================
		# 3. Collect images from valid category folders
		# Example:
		# C:\cira_batch_test\bottle\good_1.png
		# C:\cira_batch_test\bottle\bad_1.png
		# ====================================================

		# Build a reproducible image order from the selected seed.
		for category_name in sorted(os.listdir(BATCH_ROOT)):
			category_folder = os.path.join(BATCH_ROOT, category_name)

			if not os.path.isdir(category_folder):
				continue

			if category_name not in VALID_CATEGORIES:
				continue

			category_images = []

			for file_name in sorted(os.listdir(category_folder)):
				file_path = os.path.join(category_folder, file_name)

				if not os.path.isfile(file_path):
					continue

				if not file_name.lower().endswith(VALID_IMAGE_EXTENSIONS):
					continue

				category_images.append((file_path, category_name))

			# Shuffle within each category, the same seed reproduces the same order.
			rng = random.Random(SHUFFLE_SEED)
			rng.shuffle(category_images)

			image_list.extend(category_images)

		# ====================================================
		# 4. Read current index
		# ====================================================

		if not os.path.exists(INDEX_FILE):
			current_index = 0
		else:
			try:
				with open(INDEX_FILE, "r") as f:
					text = f.read().strip()
					current_index = int(text) if text != "" else 0
			except:
				current_index = 0

		# ====================================================
		# 5. No image found
		# ====================================================

		if len(image_list) == 0:
			try:
				folders_found = os.listdir(BATCH_ROOT)
			except:
				folders_found = []

			out = {
				"status": "ERROR",
				"have_img": False,
				"message": "No valid image found in category folders.",
				"batch_root": BATCH_ROOT,
				"folders_found": folders_found,
				"valid_categories": VALID_CATEGORIES,
				"image_count": 0,
				"result_text": "No valid image found in category folders.",
				"led_status": 0,
				"image_path": ""
			}

		# ====================================================
		# 6. Finished all images
		# Stop automatically when all images are finished.
		# ====================================================

		elif current_index >= len(image_list):
			out = {
				"status": "COMPLETED",
				"have_img": False,
				"message": "Batch testing completed. All images processed.",
				"result_text": "Batch testing completed. All images processed.",
				"led_status": 1,
				"image_path": "",
				"category": "",
				"mode": "completed",
				"index": current_index,
				"total": len(image_list),
				"image_count": len(image_list)
			}

		# ====================================================
		# 7. Load current image and prepare payload for Flask
		# ====================================================

		else:
			current_image_path, category = image_list[current_index]
			current_image_path = current_image_path.replace("\\", "/")

			out = {
				"status": "OK",
				"have_img": True,
				"message": "Image found.",
				"image_path": current_image_path,
				"category": category,
				"mode": MODE,
				"index": current_index + 1,
				"total": len(image_list),
				"image_count": len(image_list)
			}

			# Save next index
			try:
				with open(INDEX_FILE, "w") as f:
					f.write(str(current_index + 1))
			except Exception as e:
				out = {
					"status": "ERROR",
					"have_img": False,
					"message": "Failed to write batch_index.txt.",
					"error": str(e),
					"result_text": "Failed to write batch_index.txt.",
					"led_status": 0,
					"image_path": ""
				}

# Expose the same result under the output names used by CiRA.
payload = out
output = out
result = out