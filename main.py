"""
You can run this script via 'python main.py' to start the GUI application.
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
from models.srcnn_2x.srcnn_2x import load_srcnn_2x, predict_srcnn_2x
from models.srcnn_4x.srcnn_4x import load_srcnn_4x, predict_srcnn_4x

# paths to models
SRCNN_2X_PATH = os.path.join(os.getcwd(), "models/srcnn_2x/srcnn_model_2x.pth")
SRCNN_4X_PATH = os.path.join(os.getcwd(), "models/srcnn_4x/srcnn_model_4x.pth")


class SuperResolutionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Super Resolution App")
        master.geometry("1300x600")

        self.create_widgets()
        self.scale_factor = 2

    def create_widgets(self):
        # frame for controls
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # upscale factor
        ttk.Label(control_frame, text="Upscale Factor:").pack(anchor=tk.W)
        self.upscale_var = tk.StringVar(value="2x")
        ttk.Radiobutton(control_frame, text="2x", variable=self.upscale_var, value="2x", command=self.update_scale_factor).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="4x", variable=self.upscale_var, value="4x", command=self.update_scale_factor).pack(anchor=tk.W)

        # model selection
        ttk.Label(control_frame, text="Model:").pack(anchor=tk.W, pady=(10, 0))
        self.model_var = tk.StringVar(value="srcnn")
        models = ["srcnn", "vdsr", "edsr", "srgan"]
        for model in models:
            ttk.Radiobutton(control_frame, text=model, variable=self.model_var, value=model).pack(anchor=tk.W)

        # buttons
        ttk.Button(control_frame, text="Choose Image", command=self.choose_image).pack(pady=10)
        ttk.Button(control_frame, text="Upscale", command=self.upscale_image).pack()

        # frame for images
        self.image_frame = ttk.Frame(self.master)
        self.image_frame.pack(side=tk.RIGHT, padx=10, pady=10, expand=True, fill=tk.BOTH)

        # labels for images
        self.original_label = ttk.Label(self.image_frame)
        self.original_label.pack(side=tk.LEFT, padx=5)

        self.upscaled_label = ttk.Label(self.image_frame)
        self.upscaled_label.pack(side=tk.RIGHT, padx=5)

    def update_scale_factor(self):
        self.scale_factor = 4 if self.upscale_var.get() == "4x" else 2
        if hasattr(self, 'original_image'):
            self.display_upscaled_input()

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        
        if file_path:
            self.original_image_path = file_path
            self.original_image = Image.open(file_path)
            self.display_upscaled_input()
            self.upscaled_label.config(image=None)  # clear the upscaled image

    def display_upscaled_input(self):
        # calculate new size based on scale factor
        new_size = (self.original_image.width * self.scale_factor, self.original_image.height * self.scale_factor)
        
        # upscale the image without smoothing to show pixels
        upscaled_input = self.original_image.resize(new_size, Image.NEAREST)
        
        # convert to PhotoImage and display
        photo = ImageTk.PhotoImage(upscaled_input)
        self.original_label.config(image=photo)
        self.original_label.image = photo

    def display_image(self, image, label):
        # convert to PhotoImage and display
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def upscale_image(self):
        if hasattr(self, 'original_image'):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            scale = self.scale_factor
            model_name = self.model_var.get()
            super_res_img = None

            if model_name == "srcnn":
                if scale == 2:
                    model = load_srcnn_2x(SRCNN_2X_PATH, device)
                    _, super_res_img, _, _ = predict_srcnn_2x(model, self.original_image_path, device)
                elif scale == 4:
                    model = load_srcnn_4x(SRCNN_4X_PATH, device)
                    _, super_res_img, _, _ = predict_srcnn_4x(model, self.original_image_path, device)
            elif model_name == "edsr":
                # TODO: Implement EDSR model
                pass

            if super_res_img is not None:
                self.display_image(super_res_img, self.upscaled_label)
            else:
                print("Error: super-resolved image is None.")

            print(f"Upscaling with {scale}x using {model_name} model")
        else:
            tk.messagebox.showinfo("Error", "Please choose an image first.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SuperResolutionGUI(root)
    root.mainloop()