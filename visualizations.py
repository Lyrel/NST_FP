import os
from pathlib import Path
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image
import numpy as np


# will accept evaluations as inputs
class Visualization:
    def __init__(self):
        self.folder_path = "./Experiments/experiment_1/intermediate"

    def count_images_in_folder(self, folder_path, count_subdirectories=False):
        path = Path(folder_path)
        file_count = sum(1 for doc in path.iterdir() if doc.is_file)
        return file_count

    def display_images_in_folder(self, experiment_id, save_file_name):
        folder_path = f"./Experiments/experiment_{experiment_id}/intermediate"
        save_path = os.path.dirname(folder_path)  # Gets the parent directory
        save_file = os.path.join(save_path, save_file_name)
        # count files in folder
        file_count = self.count_images_in_folder(folder_path)
        # Get all image paths and limit to file_count
        image_paths = natsorted(glob.glob(f"{folder_path}/*.jpg"))[:file_count]

        # Calculate number of rows needed (4 images per row)
        images_per_row = 4
        num_rows = (len(image_paths) + images_per_row - 1) // images_per_row

        # Create subplots with calculated rows
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 5 * num_rows))
        fig.subplots_adjust(wspace=0.0, hspace=0.3)

        # Flatten axes for easy iteration (handle case when only 1 row)
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        # Display images
        for idx, (ax, img_path) in enumerate(zip(axes, image_paths)):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')

            # Extract just the filename without path
            image_name = os.path.basename(img_path)

            # Add image name as text below the image
            ax.text(0.5, -0.08, image_name,
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=15,
                    wrap=True)

        # Hide any unused subplots
        for idx in range(len(image_paths), len(axes)):
            axes[idx].axis('off')

        # Save with high DPI
        plt.savefig(save_file, dpi=300, bbox_inches='tight', pad_inches=0.05)

        # Optional: Display
        # plt.show()



    def display_images_in_row(self, content_path, style_path, result_path, filename,
                              titles=['Content', 'Style', 'Stylized'],
                              text_below=['one', 'two', 'three']):

        # Load images from paths
        def load_image(image_path):
            if isinstance(image_path, str):
                img = Image.open(image_path)
                # Convert to RGB if it's RGBA
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                return np.array(img)
            else:
                # If it's already a numpy array (for backward compatibility)
                return image_path

        content_np = load_image(content_path)
        style_np = load_image(style_path)
        result_np = load_image(result_path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # List of images and titles
        images = [content_np, style_np, result_np]

        for idx, (ax, img, title, text) in enumerate(zip(axes, images, titles, text_below)):
            ax.imshow(img)

            # Add title with customization
            ax.set_title(title,
                         fontsize=14,
                         color='black',
                         pad=15)  # Padding between image and title

            # Text below image
            ax.text(0.1, -0.05, text,
                    transform=ax.transAxes,  # Use axes coordinates
                    ha='left', va='top',   # Center horizontally, top aligned
                    fontsize=10,
                    color='black')

            # Remove axes
            ax.axis('off')

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.05)  # Small gap between images

        # Add overall title if desired
        # fig.suptitle('Style Transfer Results', fontsize=18, y=1.02)

        # Save if requested
        output_dir = "./Experiments/report_images"

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Remove .jpg from filename if it was already added in the filename parameter
        if filename.endswith('.jpg'):
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = os.path.join(output_dir, filename + ".jpg")

        plt.savefig(full_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to: {full_path}")

        plt.tight_layout()
        plt.show()

    def display_images_in_row2(self, content_path, style_path, result_paths, filename,
                              titles=['Content', 'Style', 'Stylized 1', 'Stylized 2', 'Stylized 3'],
                              text_below=['', '', '', '', '']):
        """
        Display images in a 2-row layout with better alignment.
        """

        def load_image(image_path):
            if isinstance(image_path, str):
                img = Image.open(image_path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                return np.array(img)
            else:
                return image_path

        # Load images
        content_np = load_image(content_path)
        style_np = load_image(style_path)
        result_images = [load_image(path) for path in result_paths]

        # Create figure with GridSpec for precise control
        fig = plt.figure(figsize=(15, 10))

        # Define grid: 2 rows, 3 columns
        # Height ratios: top row (smaller), bottom row (larger)
        gs = fig.add_gridspec(2, 3,
                              height_ratios=[0.8, 2],
                              hspace=0.15,
                              wspace=0.05)

        # Top row: Content and Style images (centered in first two columns)
        ax_content = fig.add_subplot(gs[0, 0])
        ax_content.imshow(content_np)
        ax_content.set_title(titles[0], fontsize=12, color='black', pad=8)
        ax_content.axis('off')

        ax_style = fig.add_subplot(gs[0, 1])
        ax_style.imshow(style_np)
        ax_style.set_title(titles[1], fontsize=12, color='black', pad=8)
        ax_style.axis('off')

        # Optional: Add text below top images
        if text_below[0]:
            ax_content.text(0.5, -0.15, text_below[0],
                            transform=ax_content.transAxes,
                            ha='center', va='top',
                            fontsize=9, color='gray')

        if text_below[1]:
            ax_style.text(0.5, -0.15, text_below[1],
                          transform=ax_style.transAxes,
                          ha='center', va='top',
                          fontsize=9, color='gray')

        # Hide the third column in top row
        ax_empty = fig.add_subplot(gs[0, 2])
        ax_empty.axis('off')

        # Bottom row: Three stylized results
        for idx, (result_img, title, text) in enumerate(zip(result_images, titles[2:], text_below[2:])):
            ax_result = fig.add_subplot(gs[1, idx])
            ax_result.imshow(result_img)
            ax_result.set_title(title, fontsize=14, color='black', pad=12)

            # Add multi-line text below
            if text:
                # Split text by newline
                lines = text.split('\n')
                formatted_text = '\n'.join(lines)
                # Adjust position based on number of lines
                y_offset = -0.08 - (0.03 * (len(lines) - 1))
                ax_result.text(0.5, y_offset, formatted_text,
                               transform=ax_result.transAxes,
                               ha='center', va='top',
                               fontsize=10, color='black',
                               linespacing=1.3)

            ax_result.axis('off')

        # Save the figure
        output_dir = "./Experiments/report_images"
        os.makedirs(output_dir, exist_ok=True)

        if filename.endswith('.jpg') or filename.endswith('.png'):
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = os.path.join(output_dir, filename + ".jpg")

        plt.savefig(full_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to: {full_path}")

        plt.tight_layout()
        plt.show()

    def plot_timesteps_vs_time(self, num_steps_list, time_seconds_list, filename,
                               title="Runtime vs Number of Timesteps",
                               xlabel="Number of Timesteps",
                               ylabel="Time (seconds)"):
        """
        Create a plot showing the relationship between timesteps and runtime.

        Parameters:
        -----------
        num_steps_list : list
            List of number of timesteps values
        time_seconds_list : list
            List of corresponding time in seconds
        filename : str
            Filename to save the plot (with or without extension)
        title : str
            Title of the plot
        xlabel : str
            Label for x-axis
        ylabel : str
            Label for y-axis
        """

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data
        ax.plot(num_steps_list, time_seconds_list, 'bo-', linewidth=2, markersize=8,
                label='Runtime')

        # Add data labels
        for i, (steps, time_sec) in enumerate(zip(num_steps_list, time_seconds_list)):
            ax.annotate(f'{time_sec:.1f}s',
                        xy=(steps, time_sec),
                        xytext=(5, 20),
                        textcoords='offset points',
                        fontsize=9)

        # Customize the plot
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        # Add some style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Ensure x-axis is integer if all timesteps are integers
        if all(isinstance(x, int) for x in num_steps_list):
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Add a trend line (optional - linear regression)
        try:
            import numpy as np
            z = np.polyfit(num_steps_list, time_seconds_list, 1)
            p = np.poly1d(z)
            trend_line = p(num_steps_list)
            ax.plot(num_steps_list, trend_line, 'r--', alpha=0.7,
                    label=f'Trend (slope: {z[0]:.2f} s/step)')
            ax.legend(loc='best')
        except ImportError:
            print("NumPy not available for trend line calculation")

        # Adjust layout
        plt.tight_layout()

        # Create directory if it doesn't exist
        output_dir = "./Experiments/report_images"
        os.makedirs(output_dir, exist_ok=True)

        # Save the figure
        if not filename.endswith('.png') and not filename.endswith('.jpg') and not filename.endswith('.pdf'):
            full_path = os.path.join(output_dir, filename + ".png")
        else:
            full_path = os.path.join(output_dir, filename)

        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {full_path}")

        plt.show()

        return fig, ax






if __name__ == "__main__":

    viz = Visualization()
    # folder = "./Experiments/experiment_3/intermediate"
    # # file_count = viz.count_images_in_folder(folder)
    # viz.display_images_in_folder(3)

    img1 = "./Experiments/experiment_1/stylized_1.jpg"
    img2 = "./Experiments/experiment_2/stylized_2.jpg"
    img3 = "./Experiments/experiment_3/stylized_3.jpg"

    viz.display_images_in_row(img1, img2, img3,
                              "content_weight_impact",
                              titles=['1', '10', '50'],
                              text_below=['Content loss 3.976\nStyle loss 1.668 \nLPIPS 0.46\nOverall score 0.58',
                                          'Content loss 25.797\nStyle loss 10.279 \nLPIPS 0.49\nOverall score 0.61',
                                          'Content loss 9513.58\nStyle loss 284462.3 \nLPIPS 0.12\nOverall score 0.34'])





    # steps = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # times = [7.97, 11.24, 15.36, 20.30, 24.45, 28.19, 31.72, 36.11, 41.80]
    # viz.plot_timesteps_vs_time(steps, times, "steps_vs_runtime")


    # content_image1 = "./images/content_images/dancing.jpg"
    # content_image2 = "./images/content_images/city.jpg"
    # style_image = "./images/style_images/warhol1.png"
    # style_image2 = "./images/style_images/ryabchenko1.png"
    #
    #
    #
    # img1 = "./Experiments/experiment_1/stylized_1.jpg"
    # img2 = "./Experiments/experiment_2/stylized_2.jpg"
    # img3 = "./Experiments/experiment_3/stylized_3.jpg"

    # viz.display_images_in_row2(
    #     content_path=content_image2,
    #     style_path=style_image,
    #     result_paths=[img1, img2, img3],
    #     filename="stylization_comparison",
    #     titles=['Content Image', 'Style Image', '200 Steps', '500 Steps', '1000 Steps'],
    #     text_below=[
    #         'Content image\nfor reference',
    #         'Style image\nfor reference',
    #         'Perceptual loss: 0.234\nOverall score: 8.5\nTime: 45.2s',
    #         'Perceptual loss: 0.187\nOverall score: 8.9\nTime: 112.3s',
    #         'Perceptual loss: 0.156\nOverall score: 9.2\nTime: 224.1s'
    #     ]
    # )