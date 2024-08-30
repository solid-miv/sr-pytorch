"""
You can run this script via 'python main.py' to start the GUI application.
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from models.srcnn_2x.srcnn_2x import load_srcnn_2x, predict_srcnn_2x
from models.srcnn_4x.srcnn_4x import load_srcnn_4x, predict_srcnn_4x

# paths to models
SRCNN_2X_PATH = os.path.join(os.getcwd(), "models/srcnn_2x/srcnn_model_2x.pth")
SRCNN_4X_PATH = os.path.join(os.getcwd(), "models/srcnn_4x/srcnn_model_4x.pth")


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
            self.original_image_path = file_path
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image, self.original_label, (300, 300))

    def display_image(self, image, label, size):
        image.thumbnail(size)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    # def compare_images(img1, img2, shape1, shape2, title1="Original", title2="Super-Resolved"):
    #     plt.figure(figsize=(20,10))
    #     plt.subplot(121)
    #     plt.imshow(img1)
    #     plt.title(f"{title1}\nShape: {shape1[0]}x{shape1[1]}")
    #     plt.axis('off')
    #     plt.subplot(122)
    #     plt.imshow(img2)
    #     plt.title(f"{title2}\nShape: {shape2[0]}x{shape2[1]}")
    #     plt.axis('off')
    #     plt.show()

    def compare_images(self, img1, img2, shape1, shape2, title1="Original", title2="Super-Resolved"):
        # Convert PIL Images to numpy arrays
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.imshow(img1_array)
        plt.title(f"{title1}\nShape: {shape1[0]}x{shape1[1]}")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(img2_array)
        plt.title(f"{title2}\nShape: {shape2[0]}x{shape2[1]}")
        plt.axis('off')
        plt.show()

    # def upscale_image(self):
    #     if hasattr(self, 'original_image'):
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         print(f"Using device: {device}")

    #         scale = 2 if self.upscale_var.get() == "2x" else 4
    #         model_name = self.model_var.get()
    #         low_res_img, super_res_img, low_res_shape, super_res_shape = None, None, None, None

    #         if model_name == "srcnn":
    #             if scale == 2:
    #                 model = load_srcnn_2x(SRCNN_2X_PATH, device)
    #                 low_res_img, super_res_img, low_res_shape, super_res_shape = predict_srcnn_2x(model, self.original_image_path, device)
    #             elif scale == 4:
    #                 model = load_srcnn_4x(SRCNN_4X_PATH, device)
    #                 low_res_img, super_res_img, low_res_shape, super_res_shape = predict_srcnn_4x(model, self.original_image_path, device)
    #         elif model_name == "edsr":
    #             if scale == 2:
    #                 pass
    #                 # TODO: implement EDSR 2x
    #                 # model = load_edsr_2x(device)
    #                 # upscaled = predict_edsr_2x(model, self.original_image, device)
    #             elif scale == 4:
    #                 pass
    #                 # TODO: implement EDSR 4x
    #                 # model = load_edsr_4x(device)
    #                 # upscaled = predict_edsr_4x(model, self.original_image, device)

    #         if None not in (low_res_img, super_res_img, low_res_shape, super_res_shape):
    #             print(f"type(low_res_img): {type(low_res_img)}\ttype(super_res_img): {type(super_res_img)}")
    #             self.compare_images(low_res_img, super_res_img, low_res_shape, super_res_shape)
    #         else:
    #             print("Error: either one of the images or their shapes is None.")

    #         print(f"Upscaling with {self.upscale_var.get()} using {self.model_var.get()} model")
    #     else:
    #         tk.messagebox.showinfo("Error", "Please choose an image first.")

    def upscale_image(self):
        if hasattr(self, 'original_image'):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            scale = 2 if self.upscale_var.get() == "2x" else 4
            model_name = self.model_var.get()
            low_res_img, super_res_img, low_res_shape, super_res_shape = None, None, None, None

            if model_name == "srcnn":
                if scale == 2:
                    model = load_srcnn_2x(SRCNN_2X_PATH, device)
                    low_res_img, super_res_img, low_res_shape, super_res_shape = predict_srcnn_2x(model, self.original_image_path, device)
                elif scale == 4:
                    model = load_srcnn_4x(SRCNN_4X_PATH, device)
                    low_res_img, super_res_img, low_res_shape, super_res_shape = predict_srcnn_4x(model, self.original_image_path, device)
            elif model_name == "edsr":
                pass
                # ... (EDSR code remains the same)

            if None not in (low_res_img, super_res_img, low_res_shape, super_res_shape):
                print(f"type(low_res_img): {type(low_res_img)}\ttype(super_res_img): {type(super_res_img)}")
                self.compare_images(low_res_img, super_res_img, low_res_shape, super_res_shape)
            else:
                print("Error: either one of the images or their shapes is None.")

            print(f"Upscaling with {self.upscale_var.get()} using {self.model_var.get()} model")
        else:
            tk.messagebox.showinfo("Error", "Please choose an image first.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SuperResolutionGUI(root)
    root.mainloop()