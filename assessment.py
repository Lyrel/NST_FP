import torch
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import wasserstein_distance
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import sobel
from skimage.feature import local_binary_pattern, canny
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
import copy
import lpips



class Assessment:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Assessment starts..")

        # Initialize LPIPS model and move to device
        self.loss_fn = lpips.LPIPS(net='vgg', verbose=False)
        self.loss_fn = self.loss_fn.to(self.device)
        self.loss_fn.eval()  # Set to evaluation mode

        self.setup_transforms()
        self.metrics = {}

# MAIN FUNCTION
    def evaluate_experiment_results(self, combination, loss):
        self.extract_data(combination, loss)
        self.evauate_nst_result()
        print("Assessment completed..")
        return self.metrics

    def extract_data(self, params, loss):
        # Extract data from the folder based on the id inside the params

        self.output_dir = f"./Experiments/experiment_{params["id"]}"
        self.content_layer = params["content_layer"]
        self.style_layers = params["style_layers"]
        self.content_weight = params["content_weight"]
        self.style_weight = params["style_weight"]

        # Extract style and content loss from NST
        self.style_loss = loss[0]
        self.content_loss = loss[1]
        # Extract content, style and final stylized image
        content_img = os.path.join(self.output_dir, "content.jpg")
        style_img = os.path.join(self.output_dir, "style.jpg")
        result_img = os.path.join(self.output_dir, f"stylized_{params["id"]}.jpg")

        # Load all images
        self.content_tensor, self.content_tensor_lp, self.content_np = self.load_and_preprocess(content_img)
        self.style_tensor, self.style_tensor_lp, self.style_np = self.load_and_preprocess(style_img)
        self.result_tensor, self.result_tensor_lp, self.result_np = self.load_and_preprocess(result_img)

    def setup_transforms(self):
        """Define image transformations for VGG"""
        # ImageNet standard output (range [-2.18, 2.64])
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Transform for LPIPS (range [-1, 1])
        self.transform_lpips = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def load_and_preprocess(self, image_path):
        """Loads and preprocess image"""
        image = Image.open(image_path).convert("RGB")
        # ImageNet tensor
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        # LPIPS tensor
        tensor_lp = self.transform_lpips(image).unsqueeze(0).to(self.device)
        # Numpy array for other metrics
        image_np = np.array(image.resize((512, 512)))
        return tensor, tensor_lp, image_np

    def evauate_nst_result(self):
        """Composes dictionary with calculated metrics"""
        # ====================  PERCEPTUAL LOSS ====================
        self.metrics["loss_metrics"] = {
            "content_loss": f"{self.content_loss:.3f}",
            "style_loss": f"{self.style_loss:.3f}",
            "total_loss": f"{self.content_loss + self.style_loss:.3f}"
        }

        # ==================== 1. PERCEPTUAL METRICS ====================
        perceptual_metrics = self.perceptual_score()
        self.metrics["perceptual_metrics"] = perceptual_metrics

        # ==================== 2. PIXEL-BASED METRICS ===================
        pixel_metrics = self.pixel_based_score()
        self.metrics["pixel_metrics"] = pixel_metrics

        # ==================== 3. ARTIFACT DETECTION ====================
        artifact_metrics = self.artifact_score()
        self.metrics["artifact_metrics"] = artifact_metrics

        # ==================== 4. COHERENCE METRICS ====================
        coherence_metrics = self.coherence_score()
        self.metrics["coherence_metrics"] = coherence_metrics

        # ==================== 5. AESTHETIC QUALITY ====================
        aesthetic_metrics = self.aesthetic_score()
        self.metrics["aesthetic_metrics"] = aesthetic_metrics
        # ==================== 6. OVERALL SCORE ========================
        score = self.overall_score()
        self.metrics["overall_score"] = score

    # ==================== OVERALL SCORE CALCULATION ====================
    def overall_score(self):
        perceptual_score = self.metrics["perceptual_metrics"]["perceptual_score"]
        pixel_based_score = self.metrics["pixel_metrics"]["pixel_based_score"]
        artifact_score = self.metrics["artifact_metrics"]["artifact_score"]
        coherence_score = self.metrics["coherence_metrics"]["coherence_score"]
        aesthetic_score = self.metrics["aesthetic_metrics"]["aesthetic_score"]

        score = round(0.40 * perceptual_score +
                      0.1 * pixel_based_score +
                      0.15 * artifact_score +
                      0.25 * coherence_score +
                      0.1 * aesthetic_score, 2)
        return score

    # ==================== 1. PERCEPTUAL METRICS ====================
    def _calc_lpips(self):
        # Source lpips documentation https://pypi.org/project/lpips/
        # Ensure tensors are on correct device
        content_tensor= self.content_tensor_lp.to(self.device)
        result_tensor = self.result_tensor_lp.to(self.device)

        with torch.no_grad():
            distance = self.loss_fn(content_tensor, result_tensor).item()

        interpretation = ""
        # Needs inversion to make larger value indicate better quality
        perceptual_similarity = 1.0 - distance
        # value range Interpretation
        if perceptual_similarity < 0.2:
            interpretation = "Very different - almost fully dissimilar."
        elif perceptual_similarity < 0.4:
            interpretation = "Quite different - major perceptual changes."
        elif perceptual_similarity < 0.6:
            interpretation = "Somewhat different - clear differences."
        elif perceptual_similarity < 0.8:
            interpretation = "Moderately similar - small differences."
        elif perceptual_similarity < 0.9:
            interpretation = "Very similar - minor differences."
        elif perceptual_similarity >= 0.9:
            interpretation = "Perfect perceptual similarity."

        return round(perceptual_similarity, 2), interpretation

    def perceptual_score(self):
        lpips_score, lpips_interp = self._calc_lpips()
        return {"perceptual_score": lpips_score,
                "lpips": [lpips_score, lpips_interp]}

    # ==================== 2. PIXEL-BASED METRICS ====================
    def _ssim_content(self):
        # Source ssim documentation https://scikit-image.org/docs/0.25.x/auto_examples/transform/plot_ssim.html
        content_gray = rgb2gray(self.content_np)
        result_gray = rgb2gray(self.result_np)
        ssim_content = ssim(content_gray, result_gray, win_size=7, data_range=1.0)
        interpretation = ""
        # value range Interpretation
        if ssim_content < 0.5:
            interpretation = "Very low similarity. Content structure is heavily degraded."
        elif ssim_content < 0.7 :
            interpretation = "Low similarity. Content structure is partially lost."
        elif ssim_content < 0.85 :
            interpretation = "Moderate similarity. Content structure is preserved but degraded."
        elif ssim_content < 0.95 :
            interpretation = "Good structural preservation."
        elif ssim_content>= 0.95 :
            interpretation = "Very high structure preservation."
        return round(ssim_content, 2), interpretation

    def _psnr_content(self):
        psnr_content = psnr(self.content_np, self.result_np, data_range=255)
        norm_psnr = 0
        interpretation = ""
        # value range Interpretation
        if psnr_content < 20:
            norm_psnr = 0.1
            interpretation = "Bad pixel-level fidelity."
        elif psnr_content < 25:
            norm_psnr = 0.3
            interpretation = "Poor pixel-level fidelity."
        elif psnr_content < 30:
            norm_psnr = 0.5
            interpretation = "Fair pixel-level fidelity."
        elif psnr_content < 35:
            norm_psnr = 0.7
            interpretation = "Good pixel-level fidelity."
        elif psnr_content < 40:
            norm_psnr = 0.9
            interpretation = "Very Good pixel-level fidelity."
        elif psnr_content >= 40:
            norm_psnr = 1
            interpretation = "Excellent pixel-level fidelity."
        return round(norm_psnr, 2), interpretation

    def _earth_movement_distance(self):
        """Simple histogram visualization for quick analysis"""
        bins = 32
        emd_scores = []
        hist_style_list = []
        hist_result_list = []
        for channel_idx, (channel, color, colorcode) in enumerate(zip(range(3), ["tomato", "aquamarine", "slateblue"], ["red", "green", "blue"])):
            # Calculate histograms
            hist_style, bin_edges = np.histogram(self.style_np[:, :, channel].ravel(), bins=bins, range=(0, 255))
            hist_result, _ = np.histogram(self.result_np[:, :, channel].ravel(), bins=bins, range=(0, 255))

            hist_style_list.append(hist_style)
            hist_result_list.append(hist_result)

            # Normalize
            hist_style = hist_style / np.sum(hist_style)
            hist_result = hist_result / np.sum(hist_result)

            # Calculate EMD
            emd = wasserstein_distance(np.arange(bins), np.arange(bins), hist_style,  hist_result)
            emd_scores.append(emd)
        raw_emd = np.mean(emd_scores)
        norm_emd = 1.0 - (raw_emd/(bins - 1))
        norm_emd = np.clip(norm_emd, 0, 1)
        interpretation = ""
        # value range Interpretation
        if norm_emd < 0.5:
            interpretation = "Very poor - completely different."
        elif norm_emd < 0.7:
            interpretation = "Poor - major differences."
        elif norm_emd < 0.8:
            interpretation = "Fair - significant differences."
        elif norm_emd < 0.9:
            interpretation = "Moderate - clear differences."
        elif norm_emd < 0.95:
            interpretation = "Good - noticeable but acceptable."
        elif norm_emd >= 0.95:
            interpretation = "Very good - minor differences."

        return round(norm_emd, 2), interpretation

    def pixel_based_score(self):
        ssim_score, ssim_interp  = self._ssim_content()
        psnr_score, psnr_interp = self._psnr_content()
        emd_score, emd_interep = self._earth_movement_distance()

        score = round(0.3 * ssim_score + 0.3 * psnr_score + 0.4 * (1 - emd_score), 2)

        return { "pixel_based_score": score,
                 "ssim_content": [ssim_score, ssim_interp],
                 "psnr_content": [psnr_score, psnr_interp],
                 "emd": [emd_score, emd_interep]}

    # ==================== 3. ARTIFACT DETECTION ====================

    def _high_freq_noise(self):
        gray = rgb2gray(self.result_np)
        # High-frequency noise
        laplacian_var = ndimage.laplace(gray).var()
        # Normalize noise: lower values = better (1 = no noise)
        # Typical range: 0-5000
        norm_noise = 0
        interpretation = ""
        # value range Interpretation
        if laplacian_var < 200:
            norm_noise = 1.0
            interpretation = "Excellent. Very low noise."
        elif laplacian_var < 500 :
            norm_noise = 0.9
            interpretation = "Good. Low noise."
        elif laplacian_var < 1000 :
            norm_noise = 0.7
            interpretation = "Acceptable. Moderate noise."
        elif laplacian_var < 2000 :
            norm_noise = 0.4
            interpretation = "Poor. High noise."
        elif laplacian_var >= 2000 :
            norm_noise = max(0, 0.2 - (laplacian_var - 2000) / 10000)
            interpretation = "Very poor. Very high noise."

        return round(norm_noise, 2), interpretation

    def _unnatural_edges(self):
        # Unnatural edges
        gray = rgb2gray(self.result_np)
        edges = sobel(gray)
        edge_hist, _ = np.histogram(edges.ravel(), bins=50)
        medium_edges = np.sum((edge_hist[10:30] / np.sum(edge_hist)) > 0.05)

        # Normalize unnatural edges: lower values = better (1 = no unnatural edges)
        # Typical range: 0-50
        norm_edge = 0
        interpretation = ""

        if medium_edges < 5:
            norm_edge = 1.0
            interpretation = "Excellent. Very few unnatural edges."
        elif medium_edges < 10:
            norm_edge = 0.9
            interpretation = "Good. Few unnatural edges."
        elif medium_edges < 20:
            norm_edge = 0.7
            interpretation = "Acceptable. Moderate unnatural edges."
        elif medium_edges < 30:
            norm_edge = 0.4
            interpretation = "Poor. Many unnatural edges."
        else:
            norm_edge = max(0, 0.2 - (medium_edges - 30) / 100)
            interpretation = "Very poor. Very many unnatural edges."

        return round(norm_edge, 2), interpretation

    def _color_inconsistency(self):
        # Color inconsistency
        color_std = np.std(self.result_np, axis=(0, 1))

        # Normalize color inconsistency: lower values = better (1 = consistent)
        # color_std_values is list of [R_std, G_std, B_std]
        # Typical range: 0-100
        # if color_std is a list of three values
        max_std = max(color_std)
        norm_std = 0
        interpretation = ""
        if max_std < 20:
            norm_std = 1.0
            interpretation = "Excellent. Very consistent."
        elif max_std < 40:
            norm_std = 0.9
            interpretation = "Good. Consistent."
        elif max_std < 60:
            norm_std = 0.7
            interpretation = "Acceptable. Moderately consistent."
        elif max_std < 80:
            norm_std = 0.4
            interpretation = "Poor. Inconsistent."
        else:
            norm_std = 0.1
            interpretation = "Very poor. Very inconsistent."

        return round(norm_std, 2), interpretation

    def _unique_colors(self):
        # Quantization artifacts (posterization)
        unique_colors = len(np.unique(self.result_np.reshape(-1, 3), axis=0))
        # Normalize unique colors: higher values = better (1 = no posterization)
        # Typical range: 500-5000
        norm_colors = 0
        interpretation = ""

        if unique_colors < 500:
            norm_colors = 0.1
            interpretation = "Severe posterization."
        elif unique_colors < 1000:
            norm_colors = 0.4
            interpretation = "Moderate posterization."
        elif unique_colors < 2000:
            norm_colors = 0.7
            interpretation = "Slight posterization."
        elif unique_colors < 3000:
            norm_colors = 0.9
            interpretation = "Good color depth."
        else:
            norm_colors = 1
            interpretation = "Excellent color depth."

        return round(norm_colors, 2), interpretation

    def artifact_score(self):
        noise_score, noise_interp  = self._high_freq_noise()
        edges_score, edges_interp = self._unnatural_edges()
        colors_score, colors_interp = self._color_inconsistency()
        unique_colors_score, unique_colors_interp  = self._unique_colors()

        score = round(0.25 * noise_score + 0.25 * edges_score + 0.25 * colors_score + 0.25 * unique_colors_score, 2)
        return {
            "artifact_score": score,
            "high_freq_noise": [noise_score, noise_interp],
            "unnatural_edges": [edges_score, edges_interp],
            "color_inconsistency": [colors_score, colors_interp],
            "unique_colors": [unique_colors_score, unique_colors_interp]
        }

    # ==================== 4. COHERENCE METRICS ====================

    def _texture_coherence(self):
        # documentation https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
        # Local Binary Pattern works with single channel, therefore images myst be in greyscale
        content_gray = rgb2gray(self.content_np)
        stylized_gray = rgb2gray(self.result_np)
        # radius and sample points for texture capture
        radius = 3
        n_points = 8 * radius

        # Local Binary Pattern calculation
        lbp_content = local_binary_pattern(content_gray, n_points, radius, method="uniform")
        lbp_stylized = local_binary_pattern(stylized_gray, n_points, radius, method="uniform")

        # flattening of results to 1D array, where each bin represents a specific texture pattern
        hist_content, _ = np.histogram(lbp_content.ravel(), bins=256, range=(0, 255))
        hist_stylized, _ = np.histogram(lbp_stylized.ravel(), bins=256, range=(0, 255))
        # normalization
        hist_content = hist_content / np.sum(hist_content)
        hist_stylized = hist_stylized / np.sum(hist_stylized)

        texture_coherence = np.corrcoef(hist_content, hist_stylized)[0, 1]

        interpretation = ""
        # value range Interpretation
        if texture_coherence < 0.3:
            interpretation = "Poor texture coherence."
        elif texture_coherence < 0.5 :
            interpretation = "Moderate texture preservation."
        elif  texture_coherence < 0.7 :
            interpretation = "Good texture coherence."
        elif  texture_coherence < 0.9 :
            interpretation = "Very high texture similarity."
        elif texture_coherence >= 0.9 :
            interpretation = "Extremely high - almost identical textures."

        return round(texture_coherence, 2), interpretation

    def _edge_coherence(self):
        # documentation https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel
        content_gray = rgb2gray(self.content_np)
        stylized_gray = rgb2gray(self.result_np)

        # apply filter to detect edges
        edges_content = sobel(content_gray)
        edges_stylized = sobel(stylized_gray)

        # finds only strongest edges (over 90 threshold)
        edges_content_bin = edges_content > np.percentile(edges_content, 90)
        edges_stylized_bin = edges_stylized > np.percentile(edges_stylized, 90)

        edge_coherence = np.sum(edges_content_bin & edges_stylized_bin) / \
                         (np.sum(edges_content_bin) + 1e-8)

        interpretation = ""
        # value range Interpretation
        if edge_coherence < 0.2:
            interpretation = "Poor edge preservation."
        elif edge_coherence < 0.4 :
            interpretation = "Moderate edge preservation."
        elif  edge_coherence < 0.6 :
            interpretation = "Good edge coherence."
        elif  edge_coherence < 0.8 :
            interpretation = "Very good."
        elif edge_coherence >= 0.8 :
            interpretation = "Excellent."

        return round(edge_coherence, 2), interpretation

    def coherence_score(self):
        tex_score, texture_coh_interp = self._texture_coherence()
        edge_score, edge_coh_interp  = self._edge_coherence()

        score = round(0.4 * tex_score + 0.6 * edge_score, 2)
        return {
            "coherence_score": score,
            "texture_coherence": [tex_score, texture_coh_interp],
            "edge_coherence": [edge_score, edge_coh_interp]
        }
    # ==================== 5. AESTHETIC QUALITY ====================
    def _colorfulnes(self):
        # based on article: https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
        # and stackOverflow discussion: https://stackoverflow.com/questions/73478461/computing-colorfulness-of-an-image-in-python-fast

        # Extract channels
        R, G, B = self.result_np[:, :, 0], self.result_np[:, :, 1], self.result_np[:, :, 2]

        # Compute color oppositions
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)

        # Calculate statistics
        rgMean, rgStd = np.mean(rg), np.std(rg)
        ybMean, ybStd = np.mean(yb), np.std(yb)

        # Combine
        stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))
        colorfulness = stdRoot + (0.3 * meanRoot)

        # Normalize to [0, 1] using percentile method
        if colorfulness <= 15:
            normalized = 0.0
        elif colorfulness >= 90:
            normalized = 1.0
        else:
            normalized = (colorfulness - 15) / (90 - 15)
        # Interpretation
        if colorfulness < 15:
            interpretation = "Not colorful (grayscale/sepia)."
        elif colorfulness < 33:
            interpretation = "Slightly colorful - muted palette."
        elif colorfulness < 45:
            interpretation = "Moderately colorful."
        elif colorfulness < 59:
            interpretation = "Average colorfulness - typical photo."
        elif colorfulness < 75:
            interpretation = "Quite colorful - vibrant."
        elif colorfulness < 90:
            interpretation = "Highly colorful - saturated."
        else:
            interpretation = "Extremely colorful - maximum saturation."

        return round(normalized, 2), interpretation

    def _contrast(self):
        # Contrast
        gray = rgb2gray(self.result_np)
        contrast = np.std(gray)
        # since contrast is based on balck and white,
        # its value is between 0 and 0.5
        norm_contrast = np.clip(contrast / 0.5, 0, 1)
        interpretation = ""
        # value range Interpretation
        if norm_contrast < 0.3:
            interpretation = "Very low contrast."
        elif norm_contrast < 0.4 :
            interpretation = "Low contrast."
        elif  norm_contrast < 0.6 :
            interpretation = "Good contrast."
        elif norm_contrast < 0.8 :
            interpretation = "High contrast."
        elif norm_contrast >= 0.8 :
            interpretation = "Very high contrast."

        return round(norm_contrast, 2), interpretation

    def _sharpness(self):
        # Sharpness
        # converts to [0, 1] range
        gray = rgb2gray(self.result_np)
        # applies edge detection filter
        laplacian_edge = ndimage.laplace(gray)
        # calculates variance
        sharpness = laplacian_edge.var()
        # normalization
        norm_sharpness = np.clip(sharpness / 1000, 0, 1)
        # value range Interpretation
        interpretation = ""
        if norm_sharpness < 0.2:
            interpretation = "Blurry."
        elif norm_sharpness < 0.4 :
            interpretation = "Soft."
        elif  norm_sharpness < 0.6 :
            interpretation = "Good sharpness."
        elif norm_sharpness < 0.8 :
            interpretation = "Very Good sharpness."
        elif norm_sharpness >= 0.8 :
            interpretation = "Excellent sharpness."

        return round(norm_sharpness, 2), interpretation

    def _composition(self):
        # Composition (rule of thirds approximation)
        gray = rgb2gray(self.result_np)
        height, width = gray.shape
        edges = canny(gray, sigma=2)

        third_h = height // 3
        third_w = width // 3
        edge_mask = np.zeros_like(edges)
        edge_mask[third_h:2*third_h, :] = 1
        edge_mask[:, third_w:2*third_w] = 1
        composition_score = np.sum(edges & edge_mask) / (np.sum(edges) + 1e-8)

        interpretation = ""
        # value range Interpretation
        if composition_score < 0.15:
            interpretation = "Poor Composition."
        elif composition_score < 0.25 :
            interpretation = "Fair, some alignment with guides."
        elif composition_score < 0.35 :
            interpretation = "Good."
        elif composition_score < 0.45 :
            interpretation = "Very Good, strong rule-of-third usage."
        elif  composition_score < 0.6 :
            interpretation = "Excellent."
        elif composition_score >= 0.6 :
            interpretation = "Exceptional - highly structured."
        return round(composition_score, 2), interpretation

    def aesthetic_score(self):
        colorfulness, col_interp = self._colorfulnes()
        contrast, cont_interp = self._contrast()
        sharpness, sharp_interp = self._sharpness()
        composition, comp_interp = self._composition()

        score = round(0.3 * colorfulness + 0.2 * contrast + 0.3 * sharpness + 0.2 * composition, 2)
        return {
            "aesthetic_score": score,
            "colorfulness": [colorfulness, col_interp],
            "contrast": [contrast, cont_interp],
            "sharpness": [sharpness, sharp_interp],
            "composition": [composition, comp_interp]
        }











