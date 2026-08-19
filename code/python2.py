
# CiRA result parser: validates Flask output and prepares text, LED, and image fields.
## Checkpoint paths stay in Flask/CSV logs and are not sent to the UI.

import cv2
import os
import traceback

# Debug log for troubleshooting.
ERROR_LOG = r"C:\cira_ttl_logs\python2_error.txt"

# Validate the payload and raise malformed responses instead of hiding them.
try:
	root = payload

	if not isinstance(root, dict):
		raise TypeError(
			"Expected payload dict, got {}".format(type(root).__name__)
		)

	# CiRA may wrap the Python1 + RestPutJson result inside "payload".
	if "payload" in root and isinstance(root["payload"], dict):
		root = root["payload"]

	image_index = root.get("index", "-")
	image_total = root.get("total", "-")

	if "RestPutJson" not in root:
		raise KeyError("RestPutJson field is missing.")

	res = root["RestPutJson"]

	if not isinstance(res, dict):
		raise TypeError(
			"RestPutJson must be dict, got {}".format(type(res).__name__)
		)

	# Fields required by the CiRA display layer.
	required = [
		"status",
		"category",
		"file_name",
		"image_path",
		"anomaly_score",
		"threshold",
	]

	missing = [key for key in required if key not in res]
	if missing:
		raise KeyError(
			"Missing Flask field(s): {}".format(", ".join(missing))
		)

	status = str(res["status"])
	file_name = str(res["file_name"])
	category = str(res["category"])
	image_path = str(res["image_path"])
	label = str(res.get("label", res.get("prediction", "unknown")))

	score = float(res["anomaly_score"])
	threshold = float(
		res.get("threshold", res.get("anomaly_threshold", 0.0))
	)

	is_anomaly = res.get("is_anomaly", None)
	if is_anomaly is None:
		is_anomaly = str(label).lower() == "anomaly"
	else:
		is_anomaly = bool(is_anomaly)

	# Expose checkpoint state as a boolean, keep local paths in server logs.
	checkpoint_saved = bool(res.get("checkpoint_saved", False))
	updated_memory = bool(res.get("updated_memory", False))
	memory_size = int(res.get("memory_size", 0))

	# LED mapping: gray=error, red=anomaly, green=normal.
	if status != "OK":
		led_color = "gray"
		led_status_text = "GRAY - ERROR"
	elif is_anomaly:
		led_color = "red"
		led_status_text = "RED - ANOMALY"
	else:
		led_color = "green"
		led_status_text = "GREEN - NORMAL"

	# Text shown in the CiRA result widget.
	display_text = (
		"CTTA ANOMALY DETECTION RESULT\n"
		"------------------------------\n"
		"Progress   : {} / {}\n"
		"Category   : {}\n"
		"Result     : {}\n"
		"Is Anomaly: {}\n"
		"File       : {}\n"
		"Score      : {:.4f}\n"
		"Threshold  : {:.4f}\n"
	).format(
		image_index,
		image_total,
		category,
		label,
		is_anomaly,
		file_name,
		score,
		threshold,
	)

	img = None

	if image_path:
		win_path = image_path.replace("/", "\\")

		if os.path.exists(win_path):
			img = cv2.imread(win_path)
		elif os.path.exists(image_path):
			img = cv2.imread(image_path)

	# Keep a small, stable output schema for downstream CiRA nodes.
	out = {
		"display_text": display_text,
		"led_color": led_color,
		"led_status_text": led_status_text,
		"image_index": image_index,
		"image_total": image_total,
		"file_name": file_name,
		"category": category,
		"label": label,
		"is_anomaly": is_anomaly,
		"score": score,
		"threshold": threshold,
		"image_path": image_path,
		"image_loaded": img is not None,
		"updated_memory": updated_memory,
		"memory_size": memory_size,
		"checkpoint_saved": checkpoint_saved,
	}

	# Write a simple marker after a successful parse.
	with open(ERROR_LOG, "w") as f:
		f.write(
			"OK\n"
			"index={}\n"
			"file={}\n"
			"checkpoint_saved={}\n"
			"updated_memory={}\n".format(
				image_index,
				file_name,
				checkpoint_saved,
				updated_memory,
			)
		)

	payload = out
	output = out
	result = out
	image = img

# Log unexpected errors raise.
except Exception:
	error_text = traceback.format_exc()

	try:
		with open(ERROR_LOG, "w") as f:
			f.write(error_text)
	except:
		pass

	# Raise real failure from Cira.
	raise
