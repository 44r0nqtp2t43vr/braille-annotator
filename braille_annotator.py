"""
TODO:
- white background on braille text
- add set region threshold
- add erase function
- add move function
"""


import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

selected_boxes = []
drawing = False
ix, iy = -1, -1

BRAILLE_FONT_SIZE = 100
BRAILLE_FONT = "DejaVuSans-Bold.ttf"

def char_to_braille(c):
    BRAILLE_BASE = 0x2800
    braille_map = {
        'a': 0x01, 'b': 0x03, 'c': 0x09, 'd': 0x19, 'e': 0x11,
        'f': 0x0B, 'g': 0x1B, 'h': 0x13, 'i': 0x0A, 'j': 0x1A,
        'k': 0x05, 'l': 0x07, 'm': 0x0D, 'n': 0x1D, 'o': 0x15,
        'p': 0x0F, 'q': 0x1F, 'r': 0x17, 's': 0x0E, 't': 0x1E,
        'u': 0x25, 'v': 0x27, 'w': 0x3A, 'x': 0x2D, 'y': 0x3D, 'z': 0x35,
        ' ': 0x00
    }
    return chr(BRAILLE_BASE + braille_map.get(c.lower(), 0x00))

class BrailleOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Braille OCR Tool")

        self.a4_width = 3508
        self.a4_height = 2480
        self.preview_scale = 0.2
        self.canvas_width = int(self.a4_width * self.preview_scale)
        self.canvas_height = int(self.a4_height * self.preview_scale)

        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="gray20", cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)

        self.slider = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Threshold")
        self.slider.set(128)
        self.slider.pack(side="top", fill="x", padx=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(side="top", fill="x", pady=5)

        tk.Button(btn_frame, text="Save Braille Output", command=self.save_braille_output).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Preview Braille Text", command=self.preview_braille_text).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Invert Image", command=self.invert_image).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Undo Overlay", command=self.undo_overlay).pack(side="left", padx=5)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.orig_image = None
        self.gray_image = None
        self.binary_image = None
        self.backup_image = None  # For undo
        self.tk_img = None
        self.is_inverted = False

        self.load_image()
        self.slider.configure(command=self.update_threshold)

    def load_image(self):
        selected_boxes.clear()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not path:
            self.root.quit()

        use_portrait = messagebox.askyesno("Paper Orientation", "Use portrait orientation (A4 210×297mm)?")

        if use_portrait:
            self.a4_width = 2480
            self.a4_height = 3508
        else:
            self.a4_width = 3508
            self.a4_height = 2480

        self.canvas_width = int(self.a4_width * self.preview_scale)
        self.canvas_height = int(self.a4_height * self.preview_scale)
        self.canvas.config(width=self.canvas_width, height=self.canvas_height)

        img = cv2.imread(path)
        self.orig_image = img
        self.gray_image = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2GRAY)
        self.update_threshold(self.slider.get())

    def update_threshold(self, value):
        if self.gray_image is None:
            return
        img = self.gray_image
        if self.is_inverted:
            img = cv2.bitwise_not(img)
        _, thresh = cv2.threshold(img, int(value), 255, cv2.THRESH_BINARY)
        self.binary_image = self.embed_in_a4_canvas(thresh)
        self.backup_image = self.binary_image.copy()  # Backup for undo
        self.show_image(self.binary_image)

    def embed_in_a4_canvas(self, image):
        h, w = image.shape
        scale = min(self.a4_width / w, self.a4_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        a4 = np.full((self.a4_height, self.a4_width), 255, dtype=np.uint8)
        pad_x = (self.a4_width - new_w) // 2
        pad_y = (self.a4_height - new_h) // 2
        a4[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return a4

    def show_image(self, image):
        preview = cv2.resize(image, (self.canvas_width, self.canvas_height))
        rgb = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

    def on_mouse_down(self, event):
        global ix, iy, drawing
        drawing = True
        ix = int(self.canvas.canvasx(event.x) / self.preview_scale)
        iy = int(self.canvas.canvasy(event.y) / self.preview_scale)

    def on_mouse_up(self, event):
        global drawing
        if drawing:
            drawing = False
            ex = int(self.canvas.canvasx(event.x) / self.preview_scale)
            ey = int(self.canvas.canvasy(event.y) / self.preview_scale)
            x1, y1 = min(ix, ex), min(iy, ey)
            x2, y2 = max(ix, ex), max(iy, ey)
            selected_boxes.append((x1, y1, x2, y2))
            self.canvas.create_rectangle(x1 * self.preview_scale, y1 * self.preview_scale,
                                         x2 * self.preview_scale, y2 * self.preview_scale,
                                         outline="lime", width=2)

    def invert_image(self):
        self.is_inverted = not self.is_inverted
        self.update_threshold(self.slider.get())

    def extract_text_and_braille(self):
        results = []
        for (x1, y1, x2, y2) in selected_boxes:
            roi = self.binary_image[y1:y2, x1:x2]
            text = pytesseract.image_to_string(roi, config="--psm 6").strip()
            braille = '\n'.join(
                ''.join([char_to_braille(c) for c in line])
                for line in text.splitlines()
            )
            results.append((x1, y1, x2, y2, text, braille))
        return results

    def preview_braille_text(self):
        results = self.extract_text_and_braille()
        if not results:
            return
        win = tk.Toplevel(self.root)
        win.title("Braille Preview")
        text_box = tk.Text(win, wrap="word")
        for (x1, y1, x2, y2, text, braille) in results:
            text_box.insert("end", f"Box: {(x1, y1, x2, y2)}\nText: {text}\nBraille: {braille}\n\n")
        text_box.pack(fill="both", expand=True)

        def apply():
            edited = text_box.get("1.0", "end").strip()
            lines = edited.splitlines()

            pairs = []
            current_box = None
            braille_lines = []
            reading_braille = False

            for line in lines:
                stripped = line.strip()

                if stripped.startswith("Box:"):
                    # Save previous
                    if current_box and braille_lines:
                        braille_text = "\n".join(braille_lines).strip()
                        print(f"\n[Overlaying Braille]\nBox: {current_box}\nBraille:\n{braille_text}\n{'-'*40}")
                        pairs.append((current_box, braille_text))
                    # Start new box
                    try:
                        box_str = stripped[5:].strip()
                        current_box = tuple(map(int, box_str.strip("()").split(",")))
                    except:
                        current_box = None
                    braille_lines = []
                    reading_braille = False

                elif stripped.lower().startswith("braille:"):
                    reading_braille = True
                    # Include the first braille line (after "Braille: ")
                    braille_line = line.partition("Braille:")[2].strip()
                    if braille_line:
                        braille_lines.append(braille_line)

                elif stripped.lower().startswith("text:"):
                    continue  # Optional line

                elif reading_braille and current_box:
                    braille_lines.append(line)

            # Final block
            if current_box and braille_lines:
                braille_text = "\n".join(braille_lines).strip()
                print(f"\n[Overlaying Braille]\nBox: {current_box}\nBraille:\n{braille_text}\n{'-'*40}")
                pairs.append((current_box, braille_text))

            self.overlay_braille_on_image(pairs)
            win.destroy()

        tk.Button(win, text="Apply to Canvas", command=apply).pack(pady=5)

    def overlay_braille_on_image(self, pairs):
        self.backup_image = self.binary_image.copy()
        img = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font_path = BRAILLE_FONT
        font_size = BRAILLE_FONT_SIZE
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        for (x1, y1, x2, y2), braille in pairs:
            draw.rectangle([x1, y1, x2, y2], fill="white")
            lines = braille.splitlines()
            y = y1
            for line in lines:
                draw.text((x1, y), line, font=font, fill="black")
                y += font_size + 5

        self.binary_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        self.show_image(self.binary_image)

    def undo_overlay(self):
        if self.backup_image is not None:
            self.binary_image = self.backup_image.copy()
            self.show_image(self.binary_image)
            messagebox.showinfo("Undo", "Last overlay reverted.")

    def save_braille_output(self):
        output = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2RGB)
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if save_path:
            Image.fromarray(output).save(save_path, dpi=(300, 300))
            messagebox.showinfo("Success", f"Saved to: {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrailleOCRApp(root)
    root.mainloop()
