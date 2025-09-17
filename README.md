# Cotton Leaf Disease Intelligent Diagnosis

This project provides an intelligent diagnosis framework for cotton leaf diseases based on an improved YOLO model and large language model integration.  

---

## Project Introduction
The image dataset used in this project contains **five categories of cotton leaf conditions**:  
- Early-stage Curly Leaf Disease  
- Mid-stage Curly Leaf Disease  
- Healthy  
- Root-knot Nematode Disease  
- White Mold Disease  

The model detects diseased areas on cotton leaves and provides corresponding control recommendations.  
By integrating an improved YOLO network with a fine-tuned large language model, this project achieves **accurate detection, intelligent diagnosis, and decision support** for cotton leaf diseases.  

---

## Steps to Run

### 1. Configure Environment
Create and activate a Python environment, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Prepare the cotton leaf disease dataset.


### 3. Weights
Select the appropriate weights according to your task and place them in the `./output/Qwen3-1.7B/` directory.

### 4. YOLO best.py

Choose the suitable YOLO `best.py` script for your task. This file can be replaced based on your specific requirements.


### 5. Run all.py

Execute the main script to start detection and diagnosis:
```bash
python all.py
```


## Result

<img width="554" height="260" alt="image" src="https://github.com/user-attachments/assets/97b5b3e8-c762-447e-9c4d-005e8e5f80ba" />

## Acknowledgements

Special thanks to everyone who has contributed to this project.
