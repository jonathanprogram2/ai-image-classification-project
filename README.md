# 🦁 AI Image Classification & Processing Project

This project explores how deep learning models interpret images and how that behavior changes under visual transformations. Using a pre-trained MobileNetV2 model, I classified an image of a lion and cub, visualized model attention using Grad-CAM, tested robustness through image occlusion, and applied creative filters using Python.

---

## 📁 Project Contents

| File                          | Description |
|-------------------------------|-------------|
| `base_classifier.py`          | Classifies an image using MobileNetV2 and generates a Grad-CAM heatmap. |
| `basic_filter.py`             | Applies filters (blur, sharpen, edge detection, sepia, deep-fried) to an image. |
| `requirements.txt`            | List of Python packages used in the project. |
| `lionandson.png`              | Original input image used for classification and all experiments. |
| `gradcam_heatmap.png`         | Visual overlay showing which parts of the image the model focused on. |
| `lion_occlusion_black.png`    | Image with a black box applied to the central region. |
| `lion_occlusion_blur.png`     | Image with Gaussian blur applied to the central region. |
| `lion_occlusion_pixelated.png`| Image with pixelation applied to the central region. |
| `filter_blur.png`             | Output of basic blur filter. |
| `filter_sharpen.png`          | Output of sharpen filter. |
| `filter_edges.png`            | Output of edge detection filter. |
| `filter_sepia.png`            | Output of sepia filter. |
| `filter_deepfried.png`        | Output of a creative, exaggerated “deep-fried” meme-style filter. |

---

## 🔍 What This Project Demonstrates

- 📷 **Image Classification** using MobileNetV2 (`base_classifier.py`)
- 🧠 **Grad-CAM Heatmap** visualization to highlight model attention
- ❌ **Occlusion Testing** (blackout, blur, pixelation) to test model robustness
- 🎨 **Custom Image Filters** (blur, sharpen, edges, sepia, deep-fried)
- 🤝 **AI Collaboration** with ChatGPT for code planning, debugging, and reflection

---

## 🧠 Key Learnings

- Used Grad-CAM to identify which parts of an image influence the model’s prediction.
- Experimented with occlusion to test model robustness.
- Designed custom visual filters to explore aesthetic transformations.
- Practiced using AI as a collaborative programming assistant.
