import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from region_threshold_editor import RegionThresholdEditor

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

selected_boxes = []
drawing = False
ix, iy = -1, -1

BRAILLE_FONT_SIZE = 100
BRAILLE_FONT_PADDING = 25
BRAILLE_FONT = "DejaVuSans.ttf"

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
        tk.Button(btn_frame, text="Erase Region", command=self.erase_selected_region).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Move Region", command=self.enter_move_mode).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Set Region Threshold", command=self.open_region_threshold_editor).pack(side="left", padx=5)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.orig_image = None
        self.gray_image = None
        self.binary_image = None
        self.undo_stack = []
        self.tk_img = None
        self.is_inverted = False
        self.move_mode = False
        self.move_start_box = None

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
        self.undo_stack.append(self.binary_image.copy())
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
        ex = int(self.canvas.canvasx(event.x) / self.preview_scale)
        ey = int(self.canvas.canvasy(event.y) / self.preview_scale)

        if self.move_mode and self.move_start_box:
            self.move_region(self.move_start_box, (ex, ey))
            self.move_mode = False
            self.move_start_box = None
            selected_boxes.clear()
            return

        if drawing:
            drawing = False
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
            text = pytesseract.image_to_string(roi, config="--psm 6")

            # Split and strip each line of OCR text
            text_lines = [line.strip() for line in text.splitlines() if line.strip()]

            # Convert to Braille (strip text line first to prevent trailing braille space)
            braille_lines = []
            for line in text_lines:
                braille_line = ''.join([char_to_braille(c) for c in line])
                braille_lines.append(braille_line)

            # Final Braille text
            braille_text = '\n'.join(braille_lines).strip()

            # Debug print
            print(f"[BRAILLE DEBUG]\nRaw Text: {repr(text)}\nBraille: {repr(braille_text)}\n{'-'*40}")

            results.append((x1, y1, x2, y2, text, braille_text))
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
                    if current_box and braille_lines:
                        braille_text = "\n".join(braille_lines).strip()
                        print(f"\n[Overlaying Braille]\nBox: {current_box}\nBraille:\n{braille_text}\n{'-'*40}")
                        pairs.append((current_box, braille_text))
                    try:
                        box_str = stripped[5:].strip()
                        current_box = tuple(map(int, box_str.strip("()").split(",")))
                    except:
                        current_box = None
                    braille_lines = []
                    reading_braille = False

                elif stripped.lower().startswith("braille:"):
                    reading_braille = True
                    braille_line = line.partition("Braille:")[2].strip()
                    if braille_line:
                        braille_lines.append(braille_line)

                elif stripped.lower().startswith("text:"):
                    continue

                elif reading_braille and current_box:
                    braille_lines.append(line)

            if current_box and braille_lines:
                braille_text = "\n".join(braille_lines).strip()
                print(f"\n[Overlaying Braille]\nBox: {current_box}\nBraille:\n{braille_text}\n{'-'*40}")
                pairs.append((current_box, braille_text))

            self.overlay_braille_on_image(pairs)

            # 🔻 Clear bounding boxes after applying
            selected_boxes.clear()
            self.canvas.delete("all")
            self.show_image(self.binary_image)

            win.destroy()

        tk.Button(win, text="Apply to Canvas", command=apply).pack(pady=5)

    def overlay_braille_on_image(self, pairs):
        self.undo_stack.append(self.binary_image.copy())
        img = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        font_path = BRAILLE_FONT
        font_size = BRAILLE_FONT_SIZE
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        line_spacing = 5
        padding_x = BRAILLE_FONT_PADDING
        padding_y = BRAILLE_FONT_PADDING

        for (x1, y1, x2, y2), braille in pairs:
            # Strip leading/trailing whitespace from block and lines
            stripped_lines = [line.strip() for line in braille.strip().splitlines() if line.strip()]
            y = y1

            for line in stripped_lines:
                bbox = font.getbbox(line)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                bg_x1 = x1 - padding_x
                bg_y1 = y - padding_y
                bg_x2 = x1 + text_width + padding_x
                bg_y2 = y + text_height + padding_y

                # Draw white rectangle background
                draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="white")

                # Draw cleaned Braille text
                draw.text((x1, y), line, font=font, fill="black")

                y += text_height + line_spacing

        self.binary_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        self.show_image(self.binary_image)

    def undo_overlay(self):
        if selected_boxes:
            selected_boxes.clear()
            self.show_image(self.binary_image)
            messagebox.showinfo("Undo", "Selection cleared.")
            return

        if self.undo_stack:
            self.binary_image = self.undo_stack.pop()
            self.show_image(self.binary_image)
            messagebox.showinfo("Undo", "Undid last operation.")
        else:
            messagebox.showinfo("Undo", "No more actions to undo.")

    def erase_selected_region(self):
        if not selected_boxes:
            messagebox.showinfo("Erase", "No selected region to erase.")
            return

        self.undo_stack.append(self.binary_image.copy())
        for (x1, y1, x2, y2) in selected_boxes:
            cv2.rectangle(self.binary_image, (x1, y1), (x2, y2), color=255, thickness=-1)  # White fill

        selected_boxes.clear()
        self.show_image(self.binary_image)
        messagebox.showinfo("Erase", "Selected region erased.")

    def enter_move_mode(self):
        if len(selected_boxes) != 1:
            messagebox.showinfo("Move", "Please select one region to move.")
            return
        self.move_mode = True
        self.move_start_box = selected_boxes[0]
        messagebox.showinfo("Move", "Now click where you want to move the selected region.")

    def move_region(self, box, new_point):
        self.undo_stack.append(self.binary_image.copy())

        x1, y1, x2, y2 = box
        roi = self.binary_image[y1:y2, x1:x2].copy()
        w, h = x2 - x1, y2 - y1

        # Erase original
        cv2.rectangle(self.binary_image, (x1, y1), (x2, y2), color=255, thickness=-1)

        # Compute new top-left
        new_x1 = new_point[0]
        new_y1 = new_point[1]
        new_x2 = new_x1 + w
        new_y2 = new_y1 + h

        # Bounds check
        if new_x2 > self.binary_image.shape[1] or new_y2 > self.binary_image.shape[0]:
            messagebox.showwarning("Move", "Move exceeds image bounds.")
            return

        # Paste to new position
        self.binary_image[new_y1:new_y2, new_x1:new_x2] = roi
        self.show_image(self.binary_image)

    def open_region_threshold_editor(self):
        RegionThresholdEditor(self)


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
