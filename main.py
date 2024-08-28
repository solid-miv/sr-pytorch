"""
You can run this script via 'python main.py' to start the GUI application.
"""
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class SuperResolutionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Super Resolution App")
        master.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        # frame for controls
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # upscale factor
        ttk.Label(control_frame, text="Upscale Factor:").pack(anchor=tk.W)
        self.upscale_var = tk.StringVar(value="2x")
        ttk.Radiobutton(control_frame, text="2x", variable=self.upscale_var, value="2x").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="4x", variable=self.upscale_var, value="4x").pack(anchor=tk.W)

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

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image, self.original_label, (300, 300))

    def display_image(self, image, label, size):
        image.thumbnail(size)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def upscale_image(self):
        if hasattr(self, 'original_image'):
            # TODO: add actual super resolution processing here
            # for demonstration, the image is resized using PIL
            scale = 2 if self.upscale_var.get() == "2x" else 4
            upscaled = self.original_image.resize((self.original_image.width * scale, self.original_image.height * scale))
            self.display_image(upscaled, self.upscaled_label, (300, 300))

            # print selected options (replace with actual processing later)
            print(f"Upscaling with {self.upscale_var.get()} using {self.model_var.get()} model")
        else:
            tk.messagebox.showinfo("Error", "Please choose an image first.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SuperResolutionGUI(root)
    root.mainloop()