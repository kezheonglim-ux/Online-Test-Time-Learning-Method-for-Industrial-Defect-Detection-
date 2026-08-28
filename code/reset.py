import os

INDEX_FILE = r"C:\cira_batch_test\batch_index.txt"

# Reset index to first image
folder = os.path.dirname(INDEX_FILE)
if not os.path.exists(folder):
	os.makedirs(folder)

with open(INDEX_FILE, "w") as f:
	f.write("0")

out = {
	"status": "RESET_DONE",
	"message": "Reset completed. stop.txt deleted and image index reset to 0.",
	"load_complete": True,
	"result_text": "Reset completed. Next run will start from first image.",
}

payload = out
output = out
result = out