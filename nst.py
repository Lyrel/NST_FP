import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import copy
import os
from assessment import Assessment

class NeuralStyleTransfer:
    def __init__(self, parameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imsize = 512 if torch.cuda.is_available() else 128
        torch.set_default_device(self.device)

        # Load VGG model
        self.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

        # Normalization parameters
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

        # Initialize attributes
        self.id = parameters["id"]
        self.style_weight = parameters["style_weight"]
        self.content_weight = parameters["content_weight"]
        self.num_steps = parameters["num_steps"]
        self.content_layer = parameters["content_layer"]
        self.style_layers = parameters["style_layers"]
        self.style_img = self.load_image(parameters["style_image"])
        self.content_img = self.load_image(parameters["content_image"])
        self.optimizer = parameters["optimizer"]
        self.input_img = self.content_img.clone()

        self.output = None
        self.model = None
        self.style_losses = []
        self.content_losses = []

    def create_folders(self, id):
        """Creates folders for each NST run"""
        # create main folder for whole experiment
        main_folder = f"./Experiments/experiment_{id}"
        # folder for interim results
        sub_folder = f"{main_folder}/intermediate"
        try:
            os.makedirs(main_folder, exist_ok=True)
            os.makedirs(sub_folder, exist_ok=True)
            # print(f"Main folder '{main_folder}' is ready")
        except Exception as e:
            print(f"Error creating/accessing main folder: {e}")
        return main_folder, sub_folder

    def load_image(self, image_path):
        """Loads and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        # resize image, crop it and convert to tensor
        preprocess = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor(),
        ])
        image = preprocess(image).unsqueeze(0)  # Add batch dimension
        image = image.to(self.device, torch.float)
        return image

    def save_image(self, img_tensor, filename, output_dir):
        """Saves tensor as image file"""
        image = img_tensor.clone().detach().cpu().squeeze(0)
        image = torch.clamp(image, 0, 1)
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(image)
        filepath =os.path.join(output_dir, filename)
        pil_image.save(filepath)

    @staticmethod
    def gram_matrix(input):
        """Computes Gram matrix for style representation"""
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    class _ContentLoss(nn.Module):
        """VGG layer to store content loss"""
        def __init__(self, target):
            super().__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    class _StyleLoss(nn.Module):
        """VGG layer to store style loss"""
        def __init__(self, target_feature):
            super().__init__()
            self.target = NeuralStyleTransfer.gram_matrix(target_feature).detach()

        def forward(self, input):
            G = NeuralStyleTransfer.gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    class _Normalization(nn.Module):
        """VGG layer for normalization"""
        def __init__(self, mean, std):
            super().__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std

    def _get_style_model_and_losses(self, content_layer, style_layers):
        """Builds the style transfer model with loss layers"""

        # work and modifications will continue with the copy of BGG model
        cnn = copy.deepcopy(self.cnn)
        normalization = self._Normalization(self.cnn_normalization_mean, self.cnn_normalization_std)

        self.content_losses = []
        self.style_losses = []
        model = nn.Sequential(normalization)

        layer_names = []
        i = 0
        # renames layers for consistency
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            layer_names.append(name)
            model.add_module(name, layer)

            # adding Content loss layer to the model
            if name in content_layer:
                target = model(self.content_img).detach()
                content_loss = self._ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                self.content_losses.append(content_loss)

            # adding Style loss layer to the model
            if name in style_layers:
                target_feature = model(self.style_img).detach()
                style_loss = self._StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                self.style_losses.append(style_loss)

        # model trimming since not all layers are needed for NST
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], (self._ContentLoss, self._StyleLoss)):
                break

        model = model[:(i + 1)]

        self.model = model
        return model, self.style_losses, self.content_losses


    def run_style_transfer(self):
        # folders to store NST results
        output_dir, intermed_dir = self.create_folders(self.id)

        style_loss = []
        content_loss = []
        print(f"Experiment {self.id} is running...")
        print("Building the style transfer model..")
        self._get_style_model_and_losses(self.content_layer, self.style_layers)

        # Setup for optimization
        # track gradients with respect for the input image
        self.input_img.requires_grad_(True)
        # set to evaluation mode to ensure optimization? not training
        self.model.eval()
        # disables gradient computation? makes optimization faster
        self.model.requires_grad_(False)
        self.ct_loss = 0
        self.st_loss = 0
        # start optimization based on the optimizer selected
        if self.optimizer[0].lower() == "lbfgs":
            optimizer = optim.LBFGS([self.input_img])

            run = [0]

            # closure for optimization process
            def closure():
                # disable gradient computation
                with torch.no_grad():
                    # make sure pixel values stay within [0;1] range
                    self.input_img.clamp_(0, 1)

                # clears from saved gradients
                optimizer.zero_grad()
                self.model(self.input_img)
                # calculates style and content loss
                style_score = sum(sl.loss for sl in self.style_losses) * self.style_weight
                content_score = sum(cl.loss for cl in self.content_losses) * self.content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1

                if run[0] % 50 == 0:

                    print(f"Run {run[0]}: Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}")

                    # Save intermediate image
                    with torch.no_grad():
                        img_copy = self.input_img.clone()
                        self.save_image(img_copy, f"step_{run[0]}.jpg", intermed_dir)

                style_loss.append(style_score.item())
                content_loss.append(content_score.item())

                return loss

            for step in range(self.num_steps):
                optimizer.step(closure)
                if run[0] >= self.num_steps:
                    break

            # Final correction
            with torch.no_grad():
                self.input_img.clamp_(0, 1)

        elif self.optimizer[0].lower() == "adam":
            optimizer = optim.Adam([self.input_img], lr=self.optimizer[1])
            for step in range(self.num_steps):
                optimizer.zero_grad()

                # Forward pass
                self.model(self.input_img)

                # Calculate losses
                style_score = sum(sl.loss for sl in self.style_losses) * self.style_weight
                content_score = sum(cl.loss for cl in self.content_losses) * self.content_weight
                loss = style_score + content_score

                # Backward pass
                loss.backward()

                # Update input image - THIS IS THE KEY DIFFERENCE
                optimizer.step()

                # Clamp pixel values AFTER update (important for Adam)
                with torch.no_grad():
                    self.input_img.clamp_(0, 1)

                # Progress tracking
                if (step + 1) % 50 == 0:
                    print(f"Step {step + 1}/{self.num_steps}: "
                          f"Style Loss: {style_score.item():.4f} "
                          f"Content Loss: {content_score.item():.4f}")

                    # Save intermediate image
                    with torch.no_grad():
                        img_copy = self.input_img.clone()
                        self.save_image(img_copy, f"step_{step + 1}.jpg", intermed_dir)

                # Store losses
                style_loss.append(style_score.item())
                content_loss.append(content_score.item())

        # Final correction (already done for Adam in loop, but safe to do again)
        with torch.no_grad():
            self.input_img.clamp_(0, 1)

        self.output = self.input_img

        # saves stylized result, style and content image into experiment folder
        self.save_image(self.output, f"stylized_{self.id}.jpg", output_dir)
        self.save_image(self.style_img, f"style.jpg", output_dir)
        self.save_image(self.content_img, f"content.jpg", output_dir)
        print("Neural Style Transfer finished..")
        # Return style and content loss
        return [style_loss[-1], content_loss[-1]]










