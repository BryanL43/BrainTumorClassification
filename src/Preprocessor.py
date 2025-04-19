from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Preprocessor:
    def __init__(self, 
        clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple = (8, 8),
        gauss_std_radius: int = 1,
        deblur_weight_mask: float = 0.5,
        laplacian_kernel: ImageFilter.Kernel = None
    ):
        # Adjustable image filter parameters
        self.clip_limit = clip_limit;
        self.clahe_tile_grid_size = clahe_tile_grid_size;
        self.gauss_std_radius = gauss_std_radius;
        self.deblur_weight_mask = deblur_weight_mask;
        self.laplacian_kernel = laplacian_kernel;

        self.visualization_steps = []; # Debug steps

    def __college_np_image(self, arr: np.array, title: str):
        # Clamp values between 0 and 255 & convert to 8 bit (0-255)
        arr = np.clip(arr, 0, 255).astype(np.uint8);
        self.visualization_steps.append((arr, title));
    
    def debug_steps(self):
        num_steps = len(self.visualization_steps);
        plt.figure(figsize=(4 * num_steps, 4));

        # Augment images side-by-side in a figure
        for i, (img, title) in enumerate(self.visualization_steps):
            plt.subplot(1, num_steps, i + 1);
            plt.imshow(img);
            plt.title(title);
            plt.axis('off');

        plt.tight_layout();
        plt.show();

    def __call__(self, img: Image.Image):
        self.__college_np_image(np.array(img), "Original Image");

        # Convery to NumPy BGR format for OpenCV
        img_np = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR

        # Step 1: min-max normalization
        img_np = img_np.astype(np.float32);
        img_np -= img_np.min();
        if img_np.max() != 0:
            img_np = (img_np / img_np.max()) * 255.0;
        
        img_np = img_np.astype(np.uint8);
        self.__college_np_image(img_np[:, :, ::-1], "Step 1: min-max normalized");

        # Create grayscale mask to suppress background
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Step 2: CLAHE with background masking
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.clahe_tile_grid_size);
        channels = cv2.split(img_np);
        clahe_channels = [];

        for c in channels:
            c_clahe = clahe.apply(c);
            c_result = np.where(mask > 0, c_clahe, c);  # suppress CLAHE on background
            clahe_channels.append(c_result);

        clahe_img_np = cv2.merge(clahe_channels);
        self.__college_np_image(clahe_img_np[:, :, ::-1], "Step 2: CLAHE (Masked)");

        # Step 3: Gaussian Blur (reduce noise)
        img = Image.fromarray(clahe_img_np[:, :, ::-1]);  # BGR -> RGB
        blurred = img.filter(ImageFilter.GaussianBlur(radius=self.gauss_std_radius));
        self.__college_np_image(np.array(blurred), "Step 3: Gaussian Blurred");

        # Step 4: Deblur (Unsharp Mask)
        img_np = np.array(img).astype(np.float32);
        blurred_np = np.array(blurred).astype(np.float32);
        deblurred_np = img_np + self.deblur_weight_mask * (img_np - blurred_np);
        deblurred_np = np.clip(deblurred_np, 0, 255);
        self.__college_np_image(deblurred_np, "Step 4: Deblurred Image");

        # Step 5: Laplacian Filter
        laplacian_kernel = self.laplacian_kernel;

        # Convert to 8 bit (0-255) & apply laplacian filter
        laplacian_img = Image.fromarray(deblurred_np.astype(np.uint8)).filter(laplacian_kernel);
        laplacian_np = np.array(laplacian_img).astype(np.float32);

        # Prevent negative values
        laplacian_np = np.abs(laplacian_np);
        laplacian_np -= laplacian_np.min();

        # Normalize to [0-255]
        if laplacian_np.max() != 0:
            laplacian_np = laplacian_np / laplacian_np.max() * 255.0;
        laplacian_np = np.clip(laplacian_np, 0, 255);
        self.__college_np_image(laplacian_np, "Step 5: Laplacian Output");

        # Step 6: Combine Laplacian with Deblurred Image
        enhanced_np = deblurred_np + 0.5 * laplacian_np;
        enhanced_np = np.clip(enhanced_np, 0, 255);
        self.__college_np_image(enhanced_np, "Step 6: Final Enhanced");

        # Convert back to PIL 8 bit format (0-255)
        final_img = Image.fromarray(enhanced_np.astype(np.uint8));
        return final_img;