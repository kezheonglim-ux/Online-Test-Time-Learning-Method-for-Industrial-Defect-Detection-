Main code file:
--------------
- app_ctta.py- Category-aware Flask service file. It receives image_path, category, and mode from CiRA CORE, loads the corresponding category model files, runs prediction through the CTTA detector, and returns the JSON result to CiRA CORE. It supports evaluate, monitor, and calibrate modes.
- cira_ttl_anomaly.py- Core CTTA detector file. It handles YOLO26 feature extraction, adapter processing, normal memory bank comparison, anomaly score calculation, threshold decision, and optional online update.
- auto_calibrate_threshold.py- Deployment threshold calibration. Calibrated anomaly threshold and update threshold. Trusted normal deployment images used to calibrate threshold under testing folder stored in workdir\cira_ttl_calibration
- train_rev1.5.ipynb - Offiline preparation notebook, including dataset preparation, feature extraction, memory bank construction, initial threshold calibration, evalution and export of model files
  


rev1.0
-----
- basic test-time phase flow


rev1.1
-------
- improved the test-time phase become open-ended anomaly detection

rev1.2
-------
- resolve the ultralytics load, and add few .pt save for CiRA CORE use

rev1.3
-------
- add auto calibration step

rev1.4
-------
- enable and run all 15 category to outputing the corresponding memory_bank.pt, threshold.json and ttl_adpater.pt
- improving overall category's accuracy from 62.53% to 79.81%

rev1.5
-------
- Update the comment
