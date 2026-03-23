from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import os
from PIL import Image as PILImage
from reportlab.lib.enums import TA_JUSTIFY
from visualizations import Visualization

class ExperimentationReport:
    def __init__(self, params, metrics):
        print("PDF report starts..")

        self.metrics = metrics
        self.input_data = params

        self.id = params["id"]

        self.base_path = f"./Experiments/experiment_{self.id}"

        self.interim_path = f"{self.base_path}/intermediate/"
        self.filename = f"{self.base_path}/pdfReport_{self.id}.pdf"

        # Register custom fonts
        pdfmetrics.registerFont(TTFont("garret", "fonts/Garet-Book.ttf"))

    def create_report(self):
        # Content
        documentTitle = "test - Document title"
        title = f'Neural Style Transfer results of set id: {self.id}'
        subtitle1 = "Neural Style Transfer input parameters"
        subtitle4 = "Intermediate results visualization"
        subtitle2 = "Neural Style Transfer stylized image assessment"
        subtitle3 = f"Overall score: {self.metrics["overall_score"]}"
        textLines = [
            "This report investigates the effects of varying content-weight, ",
            "style-weight, and optimizer choice on the quality and convergence ",
            "speed of Neural Style Transfer. We evaluate results using ",
            "quantitative metrics and qualitative human assessment."
        ]

        nst_data= [
            ["Parameter", "Value"],
            ["Model", "vgg19"],
            ["Parameter set ID", self.id],
            ["Content Image", self.input_data["content_image"].split('/')[-1]],
            ["Style Image", self.input_data["style_image"].split('/')[-1]],
            ["Stylized Image", f"stylized_{self.input_data["id"]}.jpg"],
            ["Content layer", self.input_data["content_layer"]],
            ["Style layers",  ", ".join(self.input_data["style_layers"])],
            ["Content weight", self.input_data["content_weight"]],
            ["Style weight", self.input_data["style_weight"]],
            ["Optimizer", "".join(self.input_data["optimizer"][0]).upper()],
            ["Number of steps", self.input_data["num_steps"]]
        ]

        styles = getSampleStyleSheet()

        assessment_data = [
            ["Metric","Score", "Interpretation"],
            ["Perceptual Loss", self.metrics["loss_metrics"]["total_loss"], ""],
            ["   Content Loss", self.metrics["loss_metrics"]["content_loss"], ""],
            ["   Style Loss", self.metrics["loss_metrics"]["style_loss"], ""],

            ["Perceptual metrics", self.metrics["perceptual_metrics"]["lpips"][0], ""],
            ["   LPIPS", self.metrics["perceptual_metrics"]["lpips"][0], self.metrics["perceptual_metrics"]["lpips"][1]],

            ["Pixel-based evaluations", self.metrics["pixel_metrics"]["pixel_based_score"]],
            ["   SSIM", self.metrics["pixel_metrics"]["ssim_content"][0], self.metrics["pixel_metrics"]["ssim_content"][1],],
            ["   PSNR", self.metrics["pixel_metrics"]["psnr_content"][0], self.metrics["pixel_metrics"]["psnr_content"][1],],
            ["   EMD", self.metrics["pixel_metrics"]["emd"][0], self.metrics["pixel_metrics"]["emd"][1],],

            ["Artifact detection", self.metrics["artifact_metrics"]["artifact_score"], ""],
            ["   High frequency noise", self.metrics["artifact_metrics"]["high_freq_noise"][0], self.metrics["artifact_metrics"]["high_freq_noise"][1]],
            ["   Unnatural edges", self.metrics["artifact_metrics"]["unnatural_edges"][0], self.metrics["artifact_metrics"]["unnatural_edges"][1]],
            ["   Color inconsistency", self.metrics["artifact_metrics"]["color_inconsistency"][0], self.metrics["artifact_metrics"]["color_inconsistency"][1]],
            ["   Unique colors", self.metrics["artifact_metrics"]["unique_colors"][0], self.metrics["artifact_metrics"]["unique_colors"][1]],

            ["Coherence metrics", self.metrics["coherence_metrics"]["coherence_score"], ""],
            ["   Texture coherence", self.metrics["coherence_metrics"]["texture_coherence"][0], self.metrics["coherence_metrics"]["texture_coherence"][1]],
            ["   Edge coherence", self.metrics["coherence_metrics"]["edge_coherence"][0], self.metrics["coherence_metrics"]["edge_coherence"][1]],

            ["Aesthetic metrics", self.metrics["aesthetic_metrics"]["aesthetic_score"], ""],
            ["   Colorfulness", self.metrics["aesthetic_metrics"]["colorfulness"][0], self.metrics["aesthetic_metrics"]["colorfulness"][1]],
            ["   Contrast", self.metrics["aesthetic_metrics"]["contrast"][0], self.metrics["aesthetic_metrics"]["contrast"][1]],
            ["   Sharpness", self.metrics["aesthetic_metrics"]["sharpness"][0], self.metrics["aesthetic_metrics"]["sharpness"][1]],
            ["   Composition", self.metrics["aesthetic_metrics"]["composition"][0], self.metrics["aesthetic_metrics"]["composition"][1]]]



        # Create document template
        doc = SimpleDocTemplate(
            self.filename,
            pagesize=A4,
            title=documentTitle,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # Get styles
        styles = getSampleStyleSheet()

        # Create custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontName="garret",
            fontSize=20,
            spaceAfter=5,
            alignment=1,  # Center aligned
            textColor=colors.black
        )

        subtitle_style = ParagraphStyle(
            "CustomSubtitle",
            parent=styles["Heading2"],
            fontName="Helvetica",
            fontSize=14,
            spaceAfter=5,
            alignment=1,  # Center aligned
            textColor=colors.black
        )

        metric_group_style = ParagraphStyle(
            "CustomSubtitle",
            parent=styles["Heading2"],
            fontName="garret",
            fontSize=12,
            spaceAfter=2,
            alignment=0,  # Center aligned
            textColor=colors.black
        )

        metric_name_style = ParagraphStyle(
            "CustomSubtitle",
            parent=styles["Heading2"],
            fontName="garret",
            fontSize=12,
            spaceAfter=2,
            alignment=0,  # Center aligned
            textColor=colors.black
        )


        metric_text_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontName="garret",
            fontSize=10,
            textColor=colors.HexColor("#333333"),  # Dark grey
            spaceAfter=2,
            alignment=TA_JUSTIFY,
            leading=12,
        )

        # Builds story (content elements)
        story = []

        # Adds horizontal line
        from reportlab.platypus.flowables import HRFlowable
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 20))

        # Adds title
        title_paragraph = Paragraph(title, title_style)
        story.append(title_paragraph)
        story.append(Spacer(1, 20))

        # Get images for display
        content_path = f"{ self.base_path}/content.jpg"
        style_path = f"{ self.base_path}/style.jpg"
        output_path = f"{ self.base_path}/stylized_{self.id}.jpg"
        # Add image
        try:
            # Load images for display
            content_image = Image(content_path, width=200, height=200)
            style_image = Image(style_path , width=200, height=200)
            output_image = Image(output_path, width=400, height=400)

            # Assign text and images to columns and rows
            # Content and style image display
            input_image_data = [
                [Paragraph("Content Image", styles["Normal"]),
                 Paragraph("Style Image", styles["Normal"])],
                [content_image, style_image]  # Row 1: Images
            ]
            image_table = Table(input_image_data, colWidths=[200, 200])
            image_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 1), (-1, 1), 5),  # Space above captions
            ]))
            # Assign text and images to columns and rows
            # Stylized result image display
            stylized_image_data = [
                [Paragraph("Stylized Image", styles["Normal"])],
                [output_image],  # Row 1: Images]
            ]
            image_table2 = Table(stylized_image_data, colWidths=[400])

            # Add images and data to the PDF page
            story.append(image_table)
            story.append(image_table2)
            story.append(Spacer(1, 5))
        except:
            print(f"Warning: Could not load images.")

        # Add subtitle
        subtitle_paragraph = Paragraph(subtitle1, subtitle_style)
        story.append(subtitle_paragraph)
        story.append(Spacer(1, 5))

        # Create and style table
        nst_parameter_table = Table(nst_data, colWidths=[150, 300])
        nst_parameter_table.hAlign = "LEFT"
        nst_parameter_table.setStyle(TableStyle([
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "garret"),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

            # Data rows
            ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),

            # Grid
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),

            # Padding
            ("PADDING", (0, 0), (-1, -1), 12),
        ]))
        # Adds table with input parameters to the report page
        story.append(nst_parameter_table)
        story.append(Spacer(1, 20))

        # Adds new page
        story.append(PageBreak())

        #========INTERIM IMAGE DISPLAY ========================
        # Add subtitle
        subtitle_paragraph = Paragraph(subtitle4, subtitle_style)
        story.append(subtitle_paragraph)
        story.append(Spacer(1, 5))
        try:
            viz = Visualization()
            steps_image_name = "step.png"
            viz.display_images_in_folder(self.id, steps_image_name)
            base_path = f"./Experiments/experiment_{self.id}"
            steps_path = f"{base_path}/{steps_image_name}"

            # Calculate proportional height based on image aspect ratio

            img = PILImage.open(steps_path)
            aspect_ratio = img.height / img.width
            max_width = 400  # or whatever fits your page layout
            proportional_height = max_width * aspect_ratio

            steps_image = Image(steps_path, width=max_width, height=proportional_height)

            input_image_data = [
                [Paragraph("Intermediate steps", styles['Normal'])],
                [steps_image],  # Row 1: Images
            ]
            image_table = Table(input_image_data, colWidths=[max_width])
            story.append(image_table)
            story.append(Spacer(1, 5))

        except Exception as e:
            print(f"Warning: Could not load image: {e}")

        story.append(PageBreak())

        #======== ASSESSMENT METRICS DISPLAY ========================
        subtitle_paragraph = Paragraph(subtitle2, subtitle_style)
        story.append(subtitle_paragraph)
        story.append(Spacer(1, 5))

        # Create and style table
        assessment_table = Table(assessment_data, colWidths=[120, 60, 300])
        assessment_table.hAlign = "LEFT"
        assessment_table.setStyle(TableStyle([
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "garret"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),

            # Data rows
            ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),

            # Alignment
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

            # Grid and borders
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),

            # Row heights and padding - ADD THESE
            ("MINROWHEIGHT", (0, 0), (-1, -1), 100),  # Minimum height for all rows

            ("PADDING", (0, 0), (-1, -1), 15),       # Increased padding

            # Text wrapping
            ("WORDWRAP", (0, 0), (-1, -1), True),    # Enable text wrapping
            ("SPLITLONGWORDS", (0, 0), (-1, -1), True),
        ]))
        story.append(assessment_table)
        story.append(Spacer(1, 5))

        subtitle_paragraph = Paragraph(subtitle3, subtitle_style)
        story.append(subtitle_paragraph)
        story.append(Spacer(1, 5))

        story.append(PageBreak())

        #========METRICS DESCRIPTION DISPLAY ========================

        descriptions = f"Metric descriptions"

        # Adds title
        description_paragraph = Paragraph(descriptions, title_style)
        story.append(description_paragraph)
        story.append(Spacer(1, 20))
        # =============LOSS METRICS==================================

        # Adds subtitle
        metric_type_6 = Paragraph("Perceptual loss metrics", metric_group_style)
        story.append(metric_type_6)
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        content_loss = Paragraph("Perceptual Content Loss", metric_name_style)
        story.append(content_loss )
        content_loss_descr =("Perceptual Content Loss measures the dissimilarity between the stylized result and "
                             "the original content image based on high-level feature representations extracted from "
                             "a pre-trained "
                             "VGG network. Unlike pixel-based metrics that compare individual pixel values, this "
                             "metric evaluates differences in semantic content, object structures, and overall "
                             "scene composition by analyzing how the images activate deep neural network "
                             "layers trained to recognize visual concepts. "
                             "It captures whether the essential 'what' of the image—the objects, shapes, and their "
                             "spatial arrangement—remains intact after style transfer. Lower values indicate better "
                             "preservation of content, while higher values indicate greater deviation from the "
                             "original."
                             "<br/><br/>"
                             "A lower value indicates that the stylized image successfully preserves the "
                             "semantic content and structural essence of the original image. Objects remain "
                             "recognizable, spatial relationships between elements are maintained, and the overall "
                             "scene composition stays true to the content "
                             "source. The style has been applied without compromising the fundamental information "
                             "that allows viewers to identify what the image depicts."
                             )
        metric_detailed = Paragraph(content_loss_descr, metric_text_style)
        story.append(metric_detailed)

        style_loss = Paragraph("Perceptual Style Loss", metric_name_style)
        story.append(style_loss)
        style_loss_descr =("Perceptual Style Loss measures the dissimilarity between the stylized result and the style "
                           "image based on texture, color, and pattern representations extracted from multiple layers "
                           "of a pre-trained VGG network. This metric evaluates how well the artistic "
                           "essence—brushstrokes, color palettes, textural qualities, and repetitive "
                           "patterns—has been transferred from the "
                           "reference style to the final image. It works by comparing the Gram matrix correlations of "
                           "feature activations across different network layers, capturing stylistic characteristics "
                           "rather than semantic content. Lower values indicate better style matching, while higher "
                           "values indicate greater deviation from the intended style."
                           "<br/><br/>"
                           "A low value indicates that the stylized image successfully captures the artistic essence "
                           "of the reference style. The color palette, textural qualities, brushstroke patterns, "
                           "and overall visual rhythm closely align with the style image, creating a cohesive "
                           "artistic interpretation that feels true to the intended aesthetic. The style transfer "
                           "has effectively extracted and applied the defining characteristics of the reference "
                           "artwork."
                           )
        metric_detailed = Paragraph(style_loss_descr, metric_text_style)
        story.append(metric_detailed)
        story.append(Spacer(1, 20))
        # Adds new page
        story.append(PageBreak())
        # =============PERCEPTUAL METRICS==================================
        metric_type_1 = Paragraph("Perceptual metrics", metric_group_style)
        story.append(metric_type_1)
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        lpips = Paragraph("LPIPS", metric_name_style)
        story.append(lpips)
        lpips_descr = ("LPIPS measures the perceptual similarity between the stylized result and the original content"
                       "image by comparing deep feature representations extracted from a pre-trained VGG network. "
                       "Unlike traditional pixel-based metrics, LPIPS aligns more closely with human visual "
                       "perception, "
                       "as it evaluates differences in higher-level image structures and patterns rather than exact "
                       "pixel values."
                       "<br/><br/>"
                       "A high normalized value indicates that the stylized image preserves the perceptual essence "
                       "of the original "
                       "content very well. The image remains easily recognizable, with minimal perceptible "
                       "differences in structure, texture, and overall composition."
                       "<br/><br/>"
                       "A low normalized value suggests that the stylized image is perceptually distant "
                       "from the content image. "
                       "This may indicate excessive distortion, loss of recognizable features, or that the "
                       "artistic style has "
                       "overwhelmed the original content to the point where the subject becomes difficult to identify.")

        # Adds subtitle
        metric_detailed = Paragraph(lpips_descr, metric_text_style)
        story.append(metric_detailed)
        story.append(Spacer(1, 20))

        # =============PIXEL-BASED METRICS==================================
        # Adds subtitle
        metric_type_2 = Paragraph("Pixel-based metrics", metric_group_style)
        story.append(metric_type_2)
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        ssim = Paragraph("SSIM", metric_name_style)
        story.append(ssim)
        ssim_descr = ("SSIM ((Structural Similarity Index))measures the structural similarity between the "
                      "stylized result and the original content image by evaluating three key components: "
                      "luminance, contrast, and structural information. Unlike simple pixel-by-pixel comparisons,"
                      " SSIM assesses how well the overall structures, patterns, and spatial relationships "
                      "from the content image are preserved in the "
                      "stylized output. This metric is particularly sensitive to distortions that affect "
                      "human perception of image quality."
                      "<br/><br/>"
                      "A high normalized value indicates that the stylized image maintains strong structural "
                      "fidelity to the original content. Edges, shapes, object boundaries, and spatial arrangements "
                      "remain clearly recognizable."
                      "<br/><br/>"
                      "A low normalized value suggests significant structural degradation has occurred during the "
                      "style transfer process. The stylized image may exhibit distorted object shapes, blurred "
                      "boundaries, collapsed details, or unnatural spatial arrangements that compromise "
                      "the recognizability of the original content.")
        metric_detailed = Paragraph(ssim_descr, metric_text_style)
        story.append(metric_detailed)

        psnr = Paragraph("PSNR", metric_name_style)
        story.append(psnr)
        psnr_descr =("PSNR measures pixel-level fidelity between the stylized result and the original content "
                     "image by calculating the ratio between the maximum possible pixel value and the mean squared "
                     "error between the two images. It provides a quantitative assessment of how closely the "
                     "stylized image matches the content image at the individual pixel level, reflecting the "
                     "presence of noise, distortion, or exactness of reproduction."
                     "<br/><br/>"
                     "A high normalized value indicates excellent pixel-level preservation, meaning the "
                     "stylized image maintains precise color values, brightness levels, and fine details "
                     "with minimal deviation from the original content. "
                     "<br/><br/>"
                     " low normalized value reveals substantial pixel-level discrepancies between the"
                     " stylized result and the content image. This may manifest as visible noise, color "
                     "distortion, loss of fine details, or general degradation in image fidelity.")
        metric_detailed = Paragraph(psnr_descr, metric_text_style)
        story.append(metric_detailed)

        emd = Paragraph("EMD", metric_name_style)
        story.append(emd)
        emd_descr =("EMD (Earth Mover's Distance) measures the similarity between the color distribution of the "
                    "stylized "
                    "result and that of the style image. It calculates the minimum amount of 'work' required "
                    "to transform one color histogram into another, effectively quantifying how well the stylized "
                    "image has captured the color palette, tonal range, and overall chromatic characteristics of "
                    "the reference style image."
                    "<br/><br/>"
                    "A high normalized value indicates that the stylized image successfully replicates the color "
                    "characteristics of the style image. The color palette, saturation levels, and tonal distribution "
                    "closely match the reference, suggesting effective transfer of the style's chromatic "
                    "essence while maintaining natural color transitions."
                    "<br/><br/>"
                    "A low normalized value reveals that the stylized image has failed to capture the color "
                    "characteristics of the style image. The color distribution may differ significantly, "
                    "manifesting as incorrect hues, mismatched saturation levels, or unnatural color transitions. "
                    "This may result from inadequate style transfer optimization or interference from the content "
                    "image's original colors overwhelming the intended style palette.")

        metric_detailed = Paragraph(emd_descr, metric_text_style)
        story.append(metric_detailed)
        story.append(Spacer(1, 20))

        # =============ARTIFACT METRICS==================================
        # Adds subtitle
        metric_type_3 = Paragraph("Artifact detection metrics", metric_group_style)
        story.append(metric_type_3)
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        noise = Paragraph("High Frequency Noise", metric_name_style)
        story.append(noise )
        hfn_descr =("High-Frequency Noise measures the presence of unwanted fine-grained artifacts and random "
                    "pixel-level variations in the stylized image. It detects rapid intensity fluctuations across "
                    "neighboring pixels that typically manifest as graininess, speckling, or texture irregularities "
                    "not present in either the original content or intended style. Such noise often arises from "
                    "optimization instabilities or over-exaggeration of fine style details during the transfer process."
                    "<br/><br/>"
                    "A high normalized value indicates that the stylized image exhibits minimal high-frequency noise, "
                    "appearing clean and smooth with natural transitions between pixels. This suggests that the style "
                    "transfer process has produced a visually pleasing result free from distracting graininess or "
                    "artificial texture artifacts."
                    "<br/><br/>"
                    "A low normalized value reveals significant noise contamination in the stylized image, "
                    "manifesting as visible grain, speckling, or unnatural textural irregularities. Such "
                    "artifacts can distract from the artistic quality of the result, create an unpolished "
                    "appearance, and may indicate suboptimal optimization parameters or convergence issues "
                    "during the style transfer process.")
        metric_detailed = Paragraph(hfn_descr, metric_text_style)
        story.append(metric_detailed)

        edges = Paragraph("Unnatural Edges", metric_name_style)
        story.append(edges)
        edges_descr =("Unnatural Edges measures the presence of artificial, jagged, or overly sharp edge patterns in "
                      "the stylized image that do not correspond to natural structural boundaries. This metric "
                      "identifies edges that appear abrupt, broken, or excessively emphasized, which often result from"
                      "optimization "
                      "artifacts, oversharpening, or the algorithm's struggle to smoothly integrate style patterns "
                      "with "
                      "content structures."
                      "<br/><br/>"
                      "A high normalized value indicates that the stylized image contains natural, well-defined edges "
                      "that follow coherent structural boundaries."
                      "<br/><br/>"
                      "A low normalized value reveals the presence of unnatural edge artifacts in the stylized image. "
                      "These may manifest as jagged outlines, broken contours, unnaturally sharp boundaries, or "
                      "fragmented edges that disrupt the visual flow. Such artifacts can make the image appear "
                      "unpolished, create visual confusion, and indicate that the style transfer process has "
                      "introduced structural inconsistencies rather than gracefully integrating style with content.")
        metric_detailed = Paragraph(edges_descr, metric_text_style)
        story.append(metric_detailed)

        color = Paragraph("Color Inconsistency", metric_name_style)
        story.append(color)
        color_descr =("Color Inconsistency measures the uniformity and coherence of color distribution across the "
                      "stylized image. It evaluates whether colors are applied consistently throughout the image "
                      "or whether there are abrupt, unnatural shifts in hue, saturation, or brightness that disrupt "
                      "visual harmony. This metric helps identify areas where the style transfer has failed to "
                      "maintain consistent color application, resulting in patchy or fragmented color regions."
                      "<br/><br/>"
                      "A high normalized value indicates that the stylized image exhibits smooth, consistent "
                      "color transitions with uniform tonal quality across the entire image. Colors flow naturally "
                      "from one region to another, creating a cohesive and visually pleasing result where the "
                      "applied style appears evenly distributed without unnatural patches or discontinuities."
                      "<br/><br/>"
                      "A low normalized value reveals significant color inconsistencies in the stylized image, "
                      "manifesting as patchy color distribution, abrupt shifts in hue or saturation, or regions "
                      "that appear disconnected from the overall color scheme. Such inconsistencies may arise "
                      "when the style transfer algorithm struggles to balance conflicting color requirements "
                      "between content and style, resulting in a fragmented appearance where different areas "
                      "of the image seem to belong to different color palettes.")
        metric_detailed = Paragraph(color_descr, metric_text_style)
        story.append(metric_detailed)

        unique = Paragraph("Unique Colors", metric_name_style)
        story.append(unique)
        unique_descr =("Unique Colors measures the color depth and richness of the stylized image by quantifying the "
                       "number of distinct RGB color values present. This metric serves as an indicator of "
                       "posterization "
                       "artifacts, where smooth gradients and subtle color variations become compressed into visible "
                       "bands or blocks of uniform color. A diverse color palette typically indicates higher image "
                       "quality and more faithful reproduction of the style's chromatic complexity."
                       "<br/><br/>"
                       "A high normalized value indicates that the stylized image contains a rich and diverse color "
                       "palette with smooth gradations and subtle variations. The image displays natural color "
                       "transitions without visible banding, preserving the full chromatic complexity of the "
                       "style while maintaining smooth tonal shifts that contribute to a professional, high-quality "
                       "appearance."
                       "<br/><br/>"
                       "A low normalized value reveals significant posterization in the stylized image, "
                       "characterized by visible banding in gradients, flat color regions where natural variation "
                       "should exist, and an overall simplified color palette. This degradation typically"
                       "occurs when the optimization process fails to preserve subtle color nuances or when "
                       "compression artifacts reduce the effective color depth, resulting in an image that "
                       "appears artificially simplified or lacking the chromatic richness of the intended style.")
        metric_detailed = Paragraph(unique_descr, metric_text_style)
        story.append(metric_detailed)

        story.append(Spacer(1, 20))

        # Adds new page
        story.append(PageBreak())
        # =============COHERENCE METRICS==================================
        # Adds subtitle
        metric_type_4 = Paragraph("Coherence metrics", metric_group_style)
        story.append(metric_type_4)
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        texture = Paragraph("Texture Coherence", metric_name_style)
        story.append(texture)
        texture_descr =("Texture Coherence (LBP - Local Binary Pattern) measures the preservation of fine-grained "
                        "textural patterns from the original content image in the stylized result. Using Local "
                        "Binary Pattern analysis, this metric evaluates how well surface textures, material "
                        "appearances, and repetitive patterns—such as fabric weaves, skin pores, foliage details, "
                        "or architectural surface details—are maintained after style transfer. It assesses whether "
                        "the stylized image retains the textural character of the original content while incorporating "
                        "the artistic style."
                        "<br/><br/>"
                        "A high normalized value indicates that the stylized image successfully preserves the textural "
                        "essence of the content image. Surface patterns, material appearances, and fine structural "
                        "details remain recognizable and natural, with textures maintaining their original character "
                        "even as colors and broader stylistic elements transform. This suggests that the style "
                        "transfer has enhanced the image without destroying its inherent textural qualities."
                        "<br/><br/>"
                        "A low normalized value reveals significant loss or distortion of the original texture "
                        "patterns. The stylized image may exhibit flattened surfaces lacking material definition, "
                        "merged or smeared textural details, or the introduction of unnatural repetitive patterns "
                        "that do not correspond to either the original content or the intended style. Such texture "
                        "degradation can make the image appear artificial, overly simplified, or visually "
                        "confusing, as recognizable surface qualities essential to the content's identity "
                        "become compromised.")
        metric_detailed = Paragraph(texture_descr, metric_text_style)
        story.append(metric_detailed)


        edge_coh = Paragraph("Edge Coherence", metric_name_style)
        story.append(edge_coh)
        edge_coh_descr =("Edge Coherence measures the preservation of structural boundaries and object outlines "
                         "from the original content image in the stylized result. This metric evaluates how well "
                         "the stylized image maintains the integrity of edges, contours, and major structural "
                         "elements such as silhouettes, object boundaries, and the separation between distinct "
                         "visual elements. It ensures that while artistic style is applied, the underlying structure "
                         "of the scene remains intact and recognizable."
                         "<br/><br/>"
                         "A high normalized value indicates that the stylized image successfully preserves the "
                         "essential structural boundaries of the content image. Object outlines remain crisp and "
                         "well-defined, major structural elements maintain their shape, and the spatial relationships "
                         "between different components of the scene remain clearly distinguishable. The style transfer "
                         "enhances the image artistically without compromising the structural clarity that makes "
                         "the content recognizable."
                         "<br/><br/>"
                         "A low normalized value reveals significant degradation or distortion of structural "
                         "boundaries in the stylized image. This may manifest as blurred or softened edges where "
                         "clear boundaries should exist, fragmented or broken object outlines, merging of distinct "
                         "objects into ambiguous shapes, or complete loss of structural definition. Such degradation "
                         "can make the content difficult to interpret, compromise object recognition, and indicate "
                         "that the style transfer process has overwhelmed the essential structural information "
                         "of the original image.")
        metric_detailed = Paragraph(edge_coh_descr, metric_text_style)
        story.append(metric_detailed)

        story.append(Spacer(1, 20))
        # Adds new page
        story.append(PageBreak())
        # =============AESTHETIC METRICS==================================
        # Adds subtitle
        metric_type_5 = Paragraph("Aesthetic Quality metrics", metric_group_style)
        story.append(metric_type_5)
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        colorfulness = Paragraph("Colorfulness", metric_name_style)
        story.append(colorfulness)
        colorfulness_descr =("Colorfulness measures the vibrancy and richness of colors present in the stylized "
                             "image. This metric evaluates the intensity and variety of chromatic content, assessing "
                             "whether the image displays a lively, engaging color palette or appears muted, "
                             "desaturated, or dull. It captures the overall chromatic energy of the result, which "
                             "significantly influences its visual appeal and artistic impact."
                             "<br/><br/>"
                             "A high normalized value indicates that the stylized image exhibits vibrant, rich "
                             "colors with good saturation and chromatic variety. The image appears lively and "
                             "engaging, with colors that feel intentional and artistically expressive. Such "
                             "images capture attention and convey the intended stylistic energy effectively."
                             "<br/><br/>"
                             "A low normalized value reveals that the stylized image lacks chromatic vibrancy, "
                             "appearing muted, washed out, or predominantly grayscale. The colors may feel flat, "
                             "uninteresting, or insufficiently expressive of the intended style. This can result "
                             "from inadequate color transfer, overly aggressive smoothing, or style images that "
                             "themselves lack strong chromatic character.")
        metric_detailed = Paragraph(colorfulness_descr, metric_text_style)
        story.append(metric_detailed)


        contrast = Paragraph("Contrast", metric_name_style)
        story.append(contrast)
        contrast_descr =("Contrast measures the dynamic range between the brightest and darkest areas of the "
                         "stylized image, evaluating the overall tonal separation and visual punch of the result. "
                         "This metric assesses whether the image maintains good separation between light and shadow "
                         "areas, creating depth, dimension, and visual interest, or whether it appears flat and "
                         "lacking in tonal variation."
                         "<br/><br/>"
                         "A high normalized value indicates that the stylized image exhibits strong tonal separation "
                         "with well-defined highlights, midtones, and shadows. The image appears dynamic and "
                         "dimensional, "
                         "with visual depth that guides the viewer's attention and enhances the perception of form "
                         "and structure."
                         "<br/><br/>"
                         "A low normalized value reveals that the stylized image has poor tonal separation, appearing "
                         "flat or washed out. The image may lack proper definition between elements, resulting in a "
                         "visually uninteresting composition where details become difficult to distinguish due to "
                         "insufficient variation in brightness levels.")
        metric_detailed = Paragraph(contrast_descr, metric_text_style)
        story.append(metric_detailed)

        sharpness = Paragraph("Sharpness", metric_name_style)
        story.append(sharpness)
        sharpness_descr =("Sharpness measures the clarity and definition of details in the stylized image, evaluating "
                          "how well edges are rendered and fine details are preserved. This metric assesses whether "
                          "the image appears crisp and well-defined or suffers from blurring, softness, or loss of "
                          "fine structural information that reduces perceived quality."
                          "<br/><br/>"
                          "A high normalized value indicates that the stylized image exhibits good clarity with "
                          "well-defined details and crisp edges. Fine textures and subtle structural elements "
                          "remain distinguishable, contributing to an overall impression of quality and precision. "
                          "The image appears professionally rendered with appropriate focus on important visual "
                          "elements."
                          "<br/><br/>"
                          "A low normalized value reveals that the stylized image suffers from blurring or excessive "
                          "softness, resulting in loss of fine detail definition. Edges may appear smeared, textures "
                          "may become indistinct, and the overall image may seem out of focus or lacking in precision. "
                          "This can result from over-smoothing during optimization or insufficient preservation of "
                          "high-frequency information during the transfer process.")
        metric_detailed = Paragraph(sharpness_descr, metric_text_style)
        story.append(metric_detailed)

        story.append(PageBreak())

        composition = Paragraph("Composition", metric_name_style)
        story.append(composition)
        composition_descr =("Composition measures how well the stylized image adheres to established visual design "
                            "principles, specifically evaluating the placement of important visual elements relative "
                            "to the rule of thirds. This metric assesses whether key features, edges, and points of "
                            "interest align with the compositional guides that typically create visually pleasing and "
                            "balanced arrangements."
                            "<br/><br/>"
                            "A high normalized value indicates that the stylized image exhibits strong compositional "
                            "structure, with important visual elements aligned with the rule of thirds intersections "
                            "and guidelines. The arrangement of subjects, edges, and points of interest creates a "
                            "balanced, visually pleasing layout that naturally guides the viewer's eye through "
                            "the image."
                            "<br/><br/>"
                            "A low normalized value reveals that the stylized image lacks strong compositional "
                            "structure, with important visual elements poorly positioned relative to established "
                            "compositional principles. The arrangement may feel unbalanced, awkward, or "
                            "unintentional, potentially distracting from the artistic quality of the result "
                            "regardless of how well style and content are otherwise preserved.")
        metric_detailed = Paragraph(composition_descr, metric_text_style)
        story.append(metric_detailed)
        story.append(Spacer(1, 20))
        # =============OVERALL SCORE==================================

        metric_type_7 = Paragraph("Overall score", metric_group_style)
        story.append(metric_type_7)
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        overall_descr =("The Overall Score provides a single comprehensive measure of stylized image quality "
                        "by combining five distinct evaluation components into a weighted average. Each component "
                        "captures a different dimension of quality, ensuring that the final score reflects both "
                        "technical fidelity and artistic merit. The score ranges from 0 to 1, where higher values "
                        "indicate superior overall quality."
                        "\n\n"
                        "The Overall Score is calculated as follows:"
                        "<br/><br/>"
                        "Score = "
                        "<br/>"
                        "0.40 × Perceptual Quality + "
                        "<br/>"
                        "0.25 × Structural Coherence + "
                        "<br/>"
                        "0.15 × Artifact Detection + "
                        "<br/>"
                        "0.10 × Pixel Accuracy + "
                        "<br/>"
                        "0.10 × Aesthetic Appeal"
                        "<br/><br/>"
                        "Each component is normalized to a 0–1 scale before weighting, with higher values always "
                        "indicating better performance.")
        metric_detailed = Paragraph(overall_descr, metric_text_style)
        story.append(metric_detailed)

        story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)
        print("PDF report completed and saved.")
