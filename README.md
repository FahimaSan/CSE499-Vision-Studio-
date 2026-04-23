# CSE499-Vision-Studio-
Human-Guided Adversarial Synthesis Benchmarking the Robustness of Stable Diffusion Models and Deepfake Detection Models
#  CSE499A Benchmarking Project: Image Realism Evaluation

##  Overview

This project evaluates the realism of AI-generated images by comparing **human-assigned scores** with **Vision-Language Model (VLM) predictions**. The goal is to analyze how closely AI aligns with human perception using statistical visualization (violin plots).

---

##  Project Pipeline

1. **Image Generation**

   * Images are generated across multiple sessions.
   * Each session stores:

     * CSV files (metadata)
     * Excel files (history/logs)
     * Output images

2. **Human Evaluation**

   * Each image is assigned a realism score from **1 to 5**.

3. **Dataset Aggregation**

   * All session data is combined into a unified dataset:

| image_path | prompt | human_score |
| ---------- | ------ | ----------- |
| img1.png   | A cat  | 4           |

4. **Model-Based Scoring (VLM)**

   * A Vision-Language Model (Qwen VLM) is used to predict realism scores.
   * Input: image + prompt
   * Output: predicted realism score (1–5)

5. **Comparison & Analysis**

   * Human vs AI scores are compared.
   * Distribution is visualized using **violin plots**.

---

##  Model Details

* Model: Qwen Vision-Language Model (VLM)
* Variants:

  * 2B (recommended for testing)
  * 4B (higher accuracy, more compute required)

---

##  Visualization

### Violin Plot Purpose

* Shows distribution of scores
* Compares:

  * Human scoring behavior
  * AI scoring behavior

### Example

* If both distributions overlap → AI aligns well with humans
* If not → AI scoring is inconsistent

---

##  Setup Instructions

### 1. Install Dependencies

```bash
pip install pandas seaborn matplotlib transformers pillow
```

---

### 2. Combine Session Data

```python
import pandas as pd
import os

folder_path = "sessions"
all_dfs = []

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder_path, file))
        all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.to_csv("final_dataset.csv", index=False)
```

---

### 3. Generate AI Scores

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen-VL")
processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL")

image = Image.open("image.png")
prompt = "Rate the realism of this image from 1 to 5."

inputs = processor(images=image, text=prompt, return_tensors="pt")
output = model.generate(**inputs)

result = processor.decode(output[0])
print(result)
```

---

### 4. Create Violin Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.violinplot(data=[human_scores, ai_scores])
plt.xticks([0,1], ["Human", "AI"])
plt.title("Human vs AI Realism Scores")
plt.show()
```

---

##  Future Work (Fine-Tuning)

* Fine-tune Qwen VLM using labeled dataset
* Improve scoring accuracy
* Use LoRA for efficient training

---

##  Project Structure

```
project/
│── sessions/
│── images/
│── final_dataset.csv
│── notebook.ipynb
│── results/
│── README.md
```

---

##  Notes

* Ensure all image paths are valid
* Scores must be within 1–5
* Use GPU for faster processing

---

##  Goal

To determine how accurately AI can replicate human judgment of image realism.

---

##  Author
Fahima Shameem
Zaki Ahmed
CSE499A Project
