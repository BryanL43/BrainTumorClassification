from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

class Preprocessor:
    def __init__(
        self, 
        gauss_std_radius: int = 1, # Gaussian blur size
        deblur_strength: float = 0.5, # Deblur strength [0.5 is standard]
        laplacian_ksize: int = 3, # Laplacian kernel size
        edge_sharpen_strength: float = 0.5 # Laplacian edge sharpening strength [0.5 is standard]
    ):
        # Adjustable image filter parameters
        self.gauss_std_radius = gauss_std_radius;
        self.deblur_strength = deblur_strength;
        self.laplacian_ksize = laplacian_ksize;
        self.edge_sharpen_strength = edge_sharpen_strength;

        self.visualization_steps = []; # Debug steps

    def __collect_np_image(self, arr: np.array, title: str):
        # Clamp values between 0 and 255 & convert to 8 bit (0-255)
        arr = np.clip(arr, 0, 255).astype(np.uint8);
        self.visualization_steps.append((arr, title));
    
    def debug_steps(self, title: str = ""):
        num_steps = len(self.visualization_steps);
        plt.figure(figsize=(4 * num_steps, 4));
        plt.axis("off");
        plt.title(title + "\n");

        # Augment images side-by-side in a figure
        for i, (img, title) in enumerate(self.visualization_steps):
            plt.subplot(1, num_steps, i + 1);
            plt.imshow(img);
            plt.title(title);
            plt.axis("off");

        plt.tight_layout();
        plt.show();

    def __call__(self, img: Image.Image):
        self.__collect_np_image(np.array(img), "Original Image");

        # Convert to double precision
        img_np = np.array(img).astype(np.float32);

        # Step 1: min-max normalization (L membership function); convert to range [0, 1]
        mn = img_np.min();
        mx = img_np.max();
        img_norm = (img_np - mn) / (mx - mn) if mx != mn else np.zeros_like(img_np);

        self.__collect_np_image((img_norm * 255).astype(np.uint8), "Step 1: min-max normalized");

        # Step 2: Gaussian Blur (reduce noise)
        img_uint8 = (img_norm * 255).astype(np.uint8); # Scale back to [0, 255] first
        img_pil = Image.fromarray(img_uint8);
        blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=self.gauss_std_radius));
        blurred_np = np.array(blurred) / 255.0;  # Convert back to [0, 1]

        self.__collect_np_image((blurred_np * 255).astype(np.uint8), "Step 2: Gaussian Blur");

        # Step 3: Deblur (Unsharp Mask)
        deblurred = img_norm + self.deblur_strength * (img_norm - blurred_np);
        deblurred = np.clip(deblurred, 0, 1); # Clamp values between 0 and 1

        self.__collect_np_image((deblurred * 255).astype(np.uint8), "Step 3: Deblurred (Sharpened)");

        # Step 4: Laplacian Edge Detection
        laplacian = np.zeros_like(deblurred, dtype=np.float32);
        for i in range(3):  # Apply laplacian filter to R, G, B channels
            lap = cv2.Laplacian((deblurred[:, :, i] * 255).astype(np.uint8), cv2.CV_32F, ksize=self.laplacian_ksize);
            lap = np.abs(lap); # Drop negative values caused by direction change
            lap -= lap.min(); # Shifts minimum value to 0
            if lap.max() != 0:
                lap = lap / lap.max();  # Normalize to [0, 1]
            laplacian[:, :, i] = lap;
        
        # INT operator (sigmoid variant); boost edges a tad bit; values are already fine-tuned
        laplacian = np.clip(laplacian, 0, 1); # Clamp values between 0 and 1

        self.__collect_np_image((laplacian * 255).astype(np.uint8), "Step 4: Laplacian Edge Detection");

        # Step 5: Edge Sharpening
        edge_sharpened = deblurred + self.edge_sharpen_strength * laplacian;
        edge_sharpened = np.clip(edge_sharpened, 0, 1); # Clamp values between 0 and 1

        self.__collect_np_image((edge_sharpened * 255).astype(np.uint8), "Step 5: Edge Sharpened");

        # Convert to PIL image for torchvision pipeline
        final_img = Image.fromarray((edge_sharpened * 255).astype(np.uint8));
        return final_img;