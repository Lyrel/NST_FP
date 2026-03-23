import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import os


class AcademicResearchGUI:
    def __init__(self):
        # use placeholder images instead of content and style images
        self.content_image_path = "images/gui_images/placeholder.png"
        self.style_image_path = "images/gui_images/placeholder.png"
        self.content_layer = None
        self.style_layers = []
        self.style_weight = 1000000  # Default value
        self.content_weight = 1      # Default value
        self.num_steps = 200
        self.optimizer = ["lbfgs"]   # Default optimizer
        self.learning_rate = 0.02    # Default learning rate for Adam
        self.app = None
        self.submitted = False

    # Sets up image display. col = 0 refers to content image, col = 1 refers to style image
    def setPreviewImage(self, filepath, col):
        try:
            img = ctk.CTkImage(dark_image=Image.open(filepath), size=(300, 300))
            show_pic = ctk.CTkLabel( self.app, image=img, text="")
            show_pic.grid(row = 1, column = col, columnspan = 3, ipady = 0, sticky = "nsew")
        except Exception as err:
            print(f"Loading image error: {err}")

    # opens dialogbox allowing file selection from computer
    def selectImage(self, col):

        filename = filedialog.askopenfilename(
            initialdir = os.getcwd(),
            title = "Select Image",
            filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("PNG images", "*.png"), ("JPG images", "*.jpg"), ("JPEG images", "*.jpeg"))
        )

        if filename:
            print(f"Selected file: {filename}")
            # based on what button was pressed treat image
            # as content or style image. If button passed 0,
            # then it is a content image, if passed 1 then this is style image
            if col == 0:
                self.content_image_path = filename
            else:
                self.style_image_path = filename

            self.setPreviewImage(filename, col)

    # Collects data provided by the user and composes a dictionary of settiengs

    def nst_set_parameters(self):
        settings = {
            "content_image": self.content_image_path,
            "style_image": self.style_image_path,
            "content_layer": self.content_layer,
            "style_layers": self.style_layers,
            'style_weight': self.style_weight,
            'content_weight': self.content_weight,
            "optimizer": self.optimizer,
            "num_steps": self.num_steps,
            "id": "GUI"
        }
        return settings
        # Function for optimizer selection
    def on_optimizer_change(self, choice):
        if choice == "lbfgs":
            self.optimizer = ["lbfgs"]
            # Hide learning rate frame
            self.learning_rate_frame.grid_remove()
        elif choice == "adam":
            self.optimizer = ["adam", self.learning_rate]
            # Show learning rate frame
            self.learning_rate_frame.grid()
            print(f"Optimizer set to Adam with learning rate: {self.learning_rate}")

    # Function for learning rate change
    def on_learning_rate_change(self, value):
        try:
            lr = float(value)
            self.learning_rate = lr
            if self.optimizer[0] == "adam":
                self.optimizer = ["adam", self.learning_rate]
            print(f"Learning rate updated to: {self.learning_rate}")
        except ValueError:
            print(f"Invalid learning rate value: {value}")

    # Creates gui window and widgets
    def gui(self):

        self.app = ctk.CTk()
        self.app.title("Academic Research GUI")
        self.app.geometry("1000x800") # with  height
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Set rows and columns of the grid - MODIFIED for tighter checkbox spacing
        self.app.grid_columnconfigure(0, weight=0)  # Fixed width for labels
        self.app.grid_columnconfigure(1, weight=0)  # No weight for checkbox columns
        self.app.grid_columnconfigure(2, weight=0)
        self.app.grid_columnconfigure(3, weight=0)
        self.app.grid_columnconfigure(4, weight=0)
        self.app.grid_columnconfigure(5, weight=0)
        self.app.grid_columnconfigure(6, weight=1)

        # Set minimum widths for columns 1-4 to keep checkboxes close together
        self.app.columnconfigure(1, minsize=100)
        self.app.columnconfigure(2, minsize=100)
        self.app.columnconfigure(3, minsize=100)
        self.app.columnconfigure(4, minsize=100)

        self.app.grid_rowconfigure(0, weight=1)
        self.app.grid_rowconfigure(1, weight=1)
        self.app.grid_rowconfigure(2, weight=1)
        self.app.grid_rowconfigure(3, weight=1)
        self.app.grid_rowconfigure(4, weight=1)
        self.app.grid_rowconfigure(5, weight=1)
        self.app.grid_rowconfigure(6, weight=1)
        self.app.grid_rowconfigure(7, weight=1)
        self.app.grid_rowconfigure(8, weight=1)
        self.app.grid_rowconfigure(9, weight=1)
        self.app.grid_rowconfigure(10, weight=1)

        # Content image label
        content_label = ctk.CTkLabel(
            self.app,
            text = "Content Image",
            font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
            text_color="#4CC9F0",
            fg_color="#2D3047"
        )
        content_label.grid(row =0, column = 0,columnspan=3, ipadx = 20, ipady = 5, padx = 5, pady = 15, sticky = "sew")


        # Style image label
        style_label = ctk.CTkLabel(
            self.app,
            text = "Style Image",
            font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
            text_color="#4CC9F0",
            fg_color="#2D3047"
        )
        style_label.grid(row =0, column = 3,columnspan=3, ipadx = 20, ipady = 5, padx = 5, pady = 15, sticky = "sew")
        #
        self.setPreviewImage(self.content_image_path, 0)
        self.setPreviewImage(self.style_image_path, 3)
        #
        # # Content and style image selection buttons
        contentButton = ctk.CTkButton( self.app, text = "Content Image", width = 50, command = lambda: self.selectImage(0))
        styleButton = ctk.CTkButton( self.app, text = "Style Image", width = 50, command = lambda: self.selectImage(3))
        contentButton.grid(row =2, column = 1, sticky = "e")
        styleButton.grid(row = 2, column = 4, sticky = "e")

        # Settings ribbon
        settings_label = ctk.CTkLabel(
            self.app,
            text = "Settings",
            font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
            text_color="#4CC9F0",
            fg_color="#2D3047"
        )
        settings_label.grid(row =3, column = 0, columnspan = 6, ipadx = 5, ipady = 5, padx = 5, sticky = "new")

        # --- Content layer selection
        contentLabel = ctk.CTkLabel(self.app, text="Content layer: ")
        contentLabel.grid(row=4, column=0, sticky="e")

        def radiobutton_event():
            option = radio_var.get()
            if option == 1:
                self.content_layer = "conv_1"
                print(f"Selected content layer: {self.content_layer}")
            elif option == 2:
                self.content_layer = "conv_4"
                print(f"Selected content layer: {self.content_layer}")
            elif option == 3:
                self.content_layer = "conv_8"
                print(f"Selected content layer: {self.content_layer}")
            else:
                print("Make a content layer selection")
                return


        radio_var = ctk.IntVar(value=0)
        radiobutton_1 = ctk.CTkRadioButton(self.app, text="Shallow",
                                                     command=radiobutton_event, variable= radio_var, value=1)
        radiobutton_2 = ctk.CTkRadioButton(self.app, text="Middle",
                                                     command=radiobutton_event, variable= radio_var, value=2)
        radiobutton_3 = ctk.CTkRadioButton(self.app, text="Deep",
                                           command=radiobutton_event, variable= radio_var, value=3)
        radiobutton_1.grid(row=4, column=1, sticky="e")
        radiobutton_2.grid(row=4, column=2, sticky="e")
        radiobutton_3.grid(row=4, column=3, sticky="e")

        # === NEW: Style Weight and Content Weight Input Fields ===
        # Style Weight Frame
        style_weight_frame = ctk.CTkFrame(self.app, fg_color="transparent")
        style_weight_frame.grid(row=4, column=5, columnspan=2, sticky="w", padx=20)

        style_weight_label = ctk.CTkLabel(
            style_weight_frame,
            text="Style Weight:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        style_weight_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")



        self.style_weight_entry = ctk.CTkEntry(
            style_weight_frame,
            placeholder_text="1000000",
            width=120,
            font=ctk.CTkFont(size=12)
        )
        self.style_weight_entry.grid(row=0, column=1, padx=5, pady=5)
        self.style_weight_entry.insert(0, "1000000")  # Set default value

        # Content Weight Frame
        content_weight_frame = ctk.CTkFrame(self.app, fg_color="transparent")
        content_weight_frame.grid(row=5, column=5, columnspan=2, sticky="w", padx=20)


        content_weight_label = ctk.CTkLabel(
            content_weight_frame,
            text="Content Weight:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        content_weight_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")


        self.content_weight_entry = ctk.CTkEntry(
            content_weight_frame,
            placeholder_text="1",
            width=120,
            font=ctk.CTkFont(size=12)
        )
        self.content_weight_entry.grid(row=0, column=1, padx=5, pady=5)
        self.content_weight_entry.insert(0, "1")  # Set default value

        # === NEW: Number of Steps Input Field ===
        num_steps_frame = ctk.CTkFrame(self.app, fg_color="transparent")
        num_steps_frame.grid(row=6, column=5, columnspan=2, sticky="w", padx=20)

        num_steps_label = ctk.CTkLabel(
            num_steps_frame,
            text="Number of Steps:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        num_steps_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.num_steps_entry = ctk.CTkEntry(
            num_steps_frame,
            placeholder_text="200",
            width=120,
            font=ctk.CTkFont(size=12)
        )
        self.num_steps_entry.grid(row=0, column=1, padx=5, pady=5)
        self.num_steps_entry.insert(0, "200")  # Set default value to 200


        # === NEW: Optimizer Selection with Conditional Learning Rate ===
        # Optimizer Selection Frame
        optimizer_frame = ctk.CTkFrame(self.app, fg_color="transparent")
        optimizer_frame.grid(row=7, column=5, columnspan=2, sticky="w", padx=20)

        optimizer_label = ctk.CTkLabel(
            optimizer_frame,
            text="Optimizer:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        optimizer_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Dropdown for optimizer selection
        self.optimizer_var = ctk.StringVar(value="lbfgs")
        optimizer_menu = ctk.CTkOptionMenu(
            optimizer_frame,
            values=["lbfgs", "adam"],
            variable=self.optimizer_var,
            command=self.on_optimizer_change,
            width=120,
            font=ctk.CTkFont(size=12)
        )
        optimizer_menu.grid(row=0, column=1, padx=5, pady=5)

        # Learning Rate Frame (initially hidden for lbfgs)
        self.learning_rate_frame = ctk.CTkFrame(self.app, fg_color="transparent")
        self.learning_rate_frame.grid(row=8, column=5, columnspan=2, sticky="w", padx=20)

        learning_rate_label = ctk.CTkLabel(
            self.learning_rate_frame,
            text="Learning Rate:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        learning_rate_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.learning_rate_entry = ctk.CTkEntry(
            self.learning_rate_frame,
            placeholder_text="0.02",
            width=120,
            font=ctk.CTkFont(size=12)
        )
        self.learning_rate_entry.grid(row=0, column=1, padx=5, pady=5)
        self.learning_rate_entry.insert(0, "0.02")  # Set default learning rate
        self.learning_rate_entry.bind('<KeyRelease>', lambda e: self.on_learning_rate_change(self.learning_rate_entry.get()))

        # Initially hide learning rate frame since default is lbfgs
        self.learning_rate_frame.grid_remove()

        # === ROW 5: TITLE AND DESCRIPTION ===
        styleLabel = ctk.CTkLabel(self.app, text="Style layers:",
                                  font=ctk.CTkFont(weight="bold"))
        styleLabel.grid(row=5, column=0, sticky="e", padx=10)

        styleDescriptionLabel = ctk.CTkLabel(self.app,
                                             text="Select one or more layers",
                                             font=ctk.CTkFont(size=12),
                                             text_color="#666666")
        styleDescriptionLabel.grid(row=6, column=0, sticky="e", padx=10)


        style_layers = [
            ("conv_1", 5, 1),
            ("conv_2", 5, 2),
            ("conv_3", 5, 3),
            ("conv_4", 5, 4),

            # Row 6, columns 1-4
            ("conv_5", 6, 1),
            ("conv_6", 6, 2),
            ("conv_7", 6, 3),
            ("conv_8", 6, 4),

            # Row 7, columns 1-4
            ("conv_9", 7, 1),
            ("conv_10", 7, 2),
            ("conv_11", 7, 3),
            ("conv_12", 7, 4),

            # Row 8, columns 1-4
            ("conv_13", 8, 1),
            ("conv_14", 8, 2),
            ("conv_15", 8, 3),
            ("conv_16", 8, 4),
            ]
        self.checkbox_vars = {}
        # Create checkboxes
        for text, row, col in style_layers:
            var = ctk.BooleanVar(value=False)
            self.checkbox_vars[text] = var

            checkbox = ctk.CTkCheckBox(
                self.app,
                text=text,
                variable=var,
                font=ctk.CTkFont(size=12)
            )
            checkbox.grid(row=row, column=col, padx=5, pady=5, sticky="w")
            # Add the button

        # collects data for settings on submit
        def print_selected_layers():
            style_layers = [
                layer_name
                for layer_name, var in self.checkbox_vars.items()
                if var.get()
            ]

            self.style_layers = style_layers

            # Get weight values from entry fields
            try:
                style_weight_value = float(self.style_weight_entry.get())
                self.style_weight = style_weight_value  # ← ASSIGNED TO CLASS ATTRIBUTE
            except ValueError:
                print(f"Invalid style weight value: {self.style_weight_entry.get()}, using default 1000000")
                self.style_weight = 1000000  # ← DEFAULT VALUE

            try:
                content_weight_value = float(self.content_weight_entry.get())
                self.content_weight = content_weight_value
            except ValueError:
                print(f"Invalid content weight value: {self.content_weight_entry.get()}, using default 1")
                self.content_weight = 1

            # Get number of steps value from entry
            try:
                num_steps_value = int(self.num_steps_entry.get())
                self.num_steps = num_steps_value
            except ValueError:
                print(f"Invalid number of steps value: {self.num_steps_entry.get()}, using default 200")
                self.num_steps = 200

            # default state not submitted
            self.submitted = True
            self.app.quit()

        ctk.CTkButton(
            self.app,
            text="Submit",
            command= print_selected_layers,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=10, column=0, columnspan=5, pady=30)

        self.app.mainloop()