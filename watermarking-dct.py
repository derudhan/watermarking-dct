import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


class DCT_Watermark:
    def __init__(self):
        self.Q = 10
        self.size = 2
        self.sig_size = 100

    @staticmethod
    def __gene_signature(wm, size):
        wm = cv2.resize(wm, (size, size))
        wm = np.where(wm < np.mean(wm), 0, 1)
        return wm

    def inner_embed(self, B: np.ndarray, signature):
        sig_size = self.sig_size
        size = self.size
        w, h = B.shape[:2]
        embed_pos = [(0, 0)]
        if w > 2 * sig_size * size:
            embed_pos.append((w - sig_size * size, 0))
        if h > 2 * sig_size * size:
            embed_pos.append((0, h - sig_size * size))
        if len(embed_pos) == 3:
            embed_pos.append((w - sig_size * size, h - sig_size * size))
        for x, y in embed_pos:
            for i in range(x, x + sig_size * size, size):
                for j in range(y, y + sig_size * size, size):
                    v = np.float32(B[i : i + size, j : j + size])
                    v = cv2.dct(v)
                    v[size - 1, size - 1] = (
                        self.Q
                        * signature[((i - x) // size) * sig_size + (j - y) // size]
                    )
                    v = cv2.idct(v)
                    maximum = max(v.flatten())
                    minimum = min(v.flatten())
                    if maximum > 255:
                        v = v - (maximum - 255)
                    if minimum < 0:
                        v = v - minimum
                    B[i : i + size, j : j + size] = v
        return B

    def embed(self, cover, wm):
        B = None
        img = None
        signature = None
        if len(cover.shape) > 2:
            img = cv2.cvtColor(cover, cv2.COLOR_BGR2YUV)
            signature = self.__gene_signature(wm, self.sig_size).flatten()
            B = img[:, :, 0]
            if len(cover.shape) > 2:
                img[:, :, 0] = self.inner_embed(B, signature)
            cover = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        else:
            cover = B
        return cover

    def inner_extract(self, B):
        sig_size = 100
        size = self.size
        ext_sig = np.zeros(sig_size**2, dtype=int)
        for i in range(0, sig_size * size, size):
            for j in range(0, sig_size * size, size):
                v = cv2.dct(np.float32(B[i : i + size, j : j + size]))
                if v[size - 1, size - 1] > self.Q / 2:
                    ext_sig[(i // size) * sig_size + j // size] = 1
        return [ext_sig]

    def extract(self, wmimg):
        B = None
        if len(wmimg.shape) > 2:
            (B, G, R) = cv2.split(cv2.cvtColor(wmimg, cv2.COLOR_BGR2YUV))
        else:
            B = wmimg
        ext_sig = self.inner_extract(B)[0]
        ext_sig = np.array(ext_sig).reshape((self.sig_size, self.sig_size))
        ext_sig = np.where(ext_sig == 1, 255, 0)
        return ext_sig


class WatermarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DCT Watermarking")
        self.dct_watermark = DCT_Watermark()

        self.cover_image = None
        self.watermark_image = None

        self.initUI()

    def initUI(self):
        self.frame = ctk.CTkFrame(self.root, border_width=2)
        self.frame.pack(padx=50, pady=50, expand=True)

        self.title_label = ctk.CTkLabel(
            self.frame, text="DCT Watermarking Application", font=("Arial", 24)
        )
        self.title_label.grid(row=0, column=0, columnspan=2, padx=10, pady=20)

        self.load_cover_btn = ctk.CTkButton(
            self.frame,
            text="Load Cover Image",
            command=self.load_cover_image,
        )
        self.load_cover_btn.grid(
            row=1,
            column=0,
            padx=10,
            pady=10,
            ipadx=5,
            ipady=5,
        )

        self.load_watermark_btn = ctk.CTkButton(
            self.frame, text="Load Watermark", command=self.load_watermark_image
        )
        self.load_watermark_btn.grid(
            row=1,
            column=1,
            padx=10,
            pady=10,
            ipadx=5,
            ipady=5,
        )

        self.embed_btn = ctk.CTkButton(
            self.frame, text="Embed Watermark", command=self.embed_watermark
        )
        self.embed_btn.grid(
            row=2,
            column=0,
            padx=10,
            pady=10,
            ipadx=5,
            ipady=5,
        )

        self.extract_btn = ctk.CTkButton(
            self.frame, text="Extract Watermark", command=self.extract_watermark
        )
        self.extract_btn.grid(
            row=2,
            column=1,
            padx=10,
            pady=10,
            ipadx=5,
            ipady=5,
        )

        self.image_label = ctk.CTkLabel(self.frame, text="")
        self.image_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def load_cover_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            self.cover_image = cv2.imread(file_path)
            self.display_image(self.cover_image)

    def load_watermark_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            self.watermark_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.watermark_image)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Resize the image to fit within the frame
        img_pil = img_pil.resize(
            (600, 400), Image.Resampling.LANCZOS
        )  # Adjust size as needed

        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.configure(image=img_tk)  # Use 'configure' instead of 'config'
        self.image_label.image = img_tk  # Keep a reference to avoid garbage collection

    def embed_watermark(self):
        if self.cover_image is None or self.watermark_image is None:
            messagebox.showerror(
                "Error", "Please load both cover and watermark images."
            )
            return

        watermarked_image = self.dct_watermark.embed(
            self.cover_image, self.watermark_image
        )
        self.display_image(watermarked_image)
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")],
        )
        if save_path:
            cv2.imwrite(save_path, watermarked_image)
            messagebox.showinfo("Success", "Watermark embedded and image saved.")

    def extract_watermark(self):
        if self.cover_image is None:
            messagebox.showerror("Error", "Please load the watermarked image.")
            return

        extracted_watermark = self.dct_watermark.extract(self.cover_image)
        extracted_watermark = np.uint8(extracted_watermark)
        self.display_image(extracted_watermark)
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")],
        )
        if save_path:
            cv2.imwrite(save_path, extracted_watermark)
            messagebox.showinfo("Success", "Watermark extracted and image saved.")


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("GhostTrain.json")

    root = ctk.CTk()
    root.geometry("1440x1080")  # Adjust window size as needed
    app = WatermarkApp(root)
    root.mainloop()
