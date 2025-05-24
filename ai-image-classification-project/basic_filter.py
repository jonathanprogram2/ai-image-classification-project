from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np



def apply_filters(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((128, 128))

        blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))
        blurred.save("filter_blur.png")

        sharpened = img_resized.filter(ImageFilter.SHARPEN)
        sharpened.save("filter_sharpen.png")

        edges = img_resized.filter(ImageFilter.FIND_EDGES)
        edges.save("filter_edges.png")

        sepia = img_resized.convert("RGB")
        sepia_data = sepia.getdata()
        new_data = []
        for r, g, b in sepia_data:
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            new_data.append((min(255, tr), min(255, tg), min(255, tb)))
        sepia.putdata(new_data)
        sepia.save("filter_sepia.png")


        # Deep Fried (saturated + noisy)
        deepfried = img_resized.convert("RGB")
        np_img = np.array(deepfried).astype(np.uint8)
        np_img = np.clip(np_img * 1.8, 0, 255)
        noise = np.random.randint(0, 50, np_img.shape, dtype='uint8')
        deepfried_array = np.clip(np_img + noise, 0, 255)
        deepfried_img = Image.fromarray(deepfried_array.astype('uint8'))
        deepfried_img.save("filter_deepfried.png")

        print("All filters applied and saved.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    apply_filters("lionandson.png")