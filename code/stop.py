import os

STOP_FILE = r"C:\cira_batch_test\stop.txt"

folder = os.path.dirname(STOP_FILE)
if not os.path.exists(folder):
	os.makedirs(folder)

with open(STOP_FILE, "w") as f:
	f.write("stop")

out = {
	"status": "STOP_REQUESTED",
	"message": "stop.txt created",
	"stop_file": STOP_FILE,
	"stop_file_created": os.path.exists(STOP_FILE)
}

payload = out
output = out
result = out