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
scale = 4
BRAILLE_CELL_WIDTH = 11*scale
BRAILLE_CELL_HEIGHT = 20*scale
BRAILLE_DOT_RADIUS = 2*scale
BRAILLE_DOT_PADDING = 1*scale

# def char_to_braille(text):
#     BRAILLE_BASE = 0x2800
#     braille_map = {
#         'a': 0x01, 'b': 0x03, 'c': 0x09, 'd': 0x19, 'e': 0x11, 'f': 0x0B, 'g': 0x1B, 'h': 0x13, 'i': 0x0A, 'j': 0x1A,
#         'k': 0x05, 'l': 0x07, 'm': 0x0D, 'n': 0x1D, 'o': 0x15, 'p': 0x0F, 'q': 0x1F, 'r': 0x17, 's': 0x0E, 't': 0x1E,
#         'u': 0x25, 'v': 0x27, 'w': 0x3A, 'x': 0x2D, 'y': 0x3D, 'z': 0x35,
#         '1': 0x01, '2': 0x03, '3': 0x09, '4': 0x19, '5': 0x11, '6': 0x0B, '7': 0x1B, '8': 0x13, '9': 0x0A, '0': 0x1A,
#         '.': 0x32, ',': 0x02, ';': 0x06, ':': 0x12, '?': 0x26, '!': 0x16, '(': 0x36, ')': 0x36, '-': 0x24,
#         ' ': 0x00
#     }
    
#     result = ""
#     for char in text:
#         if char.isalpha():  # Only process alphabet characters
#             if char.isupper():
#                 result += chr(BRAILLE_BASE + 0x20)  # Capital sign
#                 char = char.lower()
#             result += chr(BRAILLE_BASE + braille_map.get(char, 0x00))
        
#     return result

def char_to_braille(text):
    BRAILLE_BASE = 0x2800
    braille_map = {
        'a': 0x01, 'b': 0x03, 'c': 0x09, 'd': 0x19, 'e': 0x11, 'f': 0x0B, 'g': 0x1B, 'h': 0x13, 'i': 0x0A, 'j': 0x1A,
        'k': 0x05, 'l': 0x07, 'm': 0x0D, 'n': 0x1D, 'o': 0x15, 'p': 0x0F, 'q': 0x1F, 'r': 0x17, 's': 0x0E, 't': 0x1E,
        'u': 0x25, 'v': 0x27, 'w': 0x3A, 'x': 0x2D, 'y': 0x3D, 'z': 0x35,
        '1': 0x01, '2': 0x03, '3': 0x09, '4': 0x19, '5': 0x11, '6': 0x0B, '7': 0x1B, '8': 0x13, '9': 0x0A, '0': 0x1A,
        '.': 0x32, ',': 0x02, ';': 0x06, ':': 0x12, '?': 0x26, '!': 0x16, '(': 0x36, ')': 0x36, '-': 0x24,
        ' ': 0x00
    }
    
    result = ""
    capitalize_next = False
    
    for char in text:
        if char.isupper():
            if not capitalize_next and (len(result) == 0 or result[-1] == chr(BRAILLE_BASE + braille_map.get(' '))) :
                result += chr(BRAILLE_BASE + 0x20) # Capital sign
            capitalize_next = True
            char = char.lower()
        else:
            capitalize_next = False
            
        if char in braille_map:
            result += chr(BRAILLE_BASE + braille_map[char])
        
    return result

class BrailleOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Braille Annotator")

        self.a4_width = 3508
        self.a4_height = 2480
        self.preview_scale = 0.2
        self.canvas_width = int(self.a4_width * self.preview_scale)
        self.canvas_height = int(self.a4_height * self.preview_scale)

        # Main layout frames
        top_frame = tk.Frame(root)
        top_frame.pack(side="top", fill="x", padx=10, pady=5)

        canvas_frame = tk.Frame(root)
        canvas_frame.pack(side="top", fill="both", expand=True)

        # --- Top Control Bar ---
        # File Operations
        file_frame = tk.LabelFrame(top_frame, text="File", padx=5, pady=5)
        file_frame.pack(side="left", padx=5)
        tk.Button(file_frame, text="Load Image", command=self.load_image).pack(side="left")
        tk.Button(file_frame, text="Save Image", command=self.save_braille_output).pack(side="left", padx=5)

        # Edit Operations
        edit_frame = tk.LabelFrame(top_frame, text="Edit", padx=5, pady=5)
        edit_frame.pack(side="left", padx=5)
        tk.Button(edit_frame, text="Annotate Selection", command=self.preview_braille_text).pack(side="left")
        tk.Button(edit_frame, text="Erase Selection", command=self.erase_selected_region).pack(side="left", padx=5)
        tk.Button(edit_frame, text="Move Selection", command=self.enter_move_mode).pack(side="left", padx=5)
        tk.Button(edit_frame, text="Undo", command=self.undo_overlay).pack(side="left")

        # View/Image Operations
        view_frame = tk.LabelFrame(top_frame, text="Image", padx=5, pady=5)
        view_frame.pack(side="left", padx=5)
        tk.Button(view_frame, text="Invert Image", command=self.invert_image).pack(side="left")
        tk.Button(view_frame, text="Set Region Threshold", command=self.open_region_threshold_editor).pack(side="left", padx=5)

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="gray20", cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)

        self.slider = tk.Scale(canvas_frame, from_=0, to=255, orient="horizontal", label="Threshold")
        self.slider.set(128)
        self.slider.pack(side="bottom", fill="x", padx=10)

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

    def load_image(self):
        selected_boxes.clear()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not path:
            if self.orig_image is None: # Quit if no image was ever loaded
                self.root.quit()
            return

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

    def do_zoom(self, event):
        # Zoom in/out based on mouse wheel direction
        if event.delta > 0: # Zoom in
            self.zoom_level *= 1.1
        else: # Zoom out
            self.zoom_level /= 1.1

        # Limit zoom level to prevent image from becoming too small or too large
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))

        # Redraw the image with the new zoom level
        self.show_image(self.binary_image)

    def load_image(self):
        selected_boxes.clear()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not path:
            if self.orig_image is None: # Quit if no image was ever loaded
                self.root.quit()
            return

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

            # Convert to Braille
            braille_text = char_to_braille(text)

            if braille_text.strip(): # Only add if there is convertible text
                results.append((x1, y1, x2, y2, text, braille_text))
        return results

    def preview_braille_text(self):
        if not selected_boxes:
            messagebox.showinfo("Annotate", "Please select a region first.")
            return

        results = self.extract_text_and_braille()
        if not results:
            messagebox.showinfo("Annotate", "No recognizable text found in the selected region(s).")
            # Clear selection boxes that contained no text
            selected_boxes.clear()
            self.show_image(self.binary_image)
            return

        win = tk.Toplevel(self.root)
        win.title("Edit and Apply Braille")
        text_box = tk.Text(win, wrap="word", height=15, width=60)
        for (x1, y1, x2, y2, text, braille) in results:
            text_box.insert("end", f"Box: {(x1, y1, x2, y2)}\nText: {text}\nBraille: {braille}\n\n")
        text_box.pack(fill="both", expand=True, padx=5, pady=5)

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
                        pairs.append((current_box, braille_text))
                    try:
                        box_str = stripped[5:].strip()
                        current_box = tuple(map(int, box_str.strip("()",).split(",")))
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
                pairs.append((current_box, braille_text))

            self.overlay_braille_on_image(pairs)

            selected_boxes.clear()
            self.show_image(self.binary_image)

            win.destroy()

        tk.Button(win, text="Apply to Image", command=apply).pack(pady=10)

    def overlay_braille_on_image(self, pairs):
        if not pairs:
            return
        self.undo_stack.append(self.binary_image.copy())
        img = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        for (x1, y1, x2, y2), braille in pairs:
            # Erase the original text region by drawing a white rectangle
            draw.rectangle([x1, y1, x2, y2], fill="white")

            stripped_lines = [line.strip() for line in braille.strip().splitlines() if line.strip()]
            current_y = y1 + BRAILLE_DOT_PADDING

            for line in stripped_lines:
                current_x = x1 + BRAILLE_DOT_PADDING
                for char_code in line:
                    self.draw_braille_dots(draw, current_x, current_y, ord(char_code))
                    current_x += BRAILLE_CELL_WIDTH + BRAILLE_DOT_PADDING
                current_y += BRAILLE_CELL_HEIGHT + BRAILLE_DOT_PADDING

        self.binary_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        self.show_image(self.binary_image)

    def draw_braille_dots(self, draw, x_offset, y_offset, braille_unicode):
        # Braille dot patterns (relative to BRAILLE_BASE 0x2800)
        # Each bit corresponds to a dot position
        dot_patterns = {
            0x01: (0, 0), 0x02: (0, 1), 0x04: (0, 2),  # Column 1 (left)
            0x08: (1, 0), 0x10: (1, 1), 0x20: (1, 2),  # Column 2 (right)
            0x40: (0, 3), 0x80: (1, 3)  # Dots 7 and 8 (bottom)
        }

        # Convert Unicode to relative braille pattern
        braille_pattern = braille_unicode - 0x2800

        for bit, (col, row) in dot_patterns.items():
            if (braille_pattern & bit) == bit:
                center_x = x_offset + col * (BRAILLE_CELL_WIDTH // 2) + BRAILLE_DOT_RADIUS + BRAILLE_DOT_PADDING
                center_y = y_offset + row * (BRAILLE_CELL_HEIGHT // 3) + BRAILLE_DOT_RADIUS + BRAILLE_DOT_PADDING
                
                # Draw filled circle
                draw.ellipse(
                    (center_x - BRAILLE_DOT_RADIUS,
                     center_y - BRAILLE_DOT_RADIUS,
                     center_x + BRAILLE_DOT_RADIUS,
                     center_y + BRAILLE_DOT_RADIUS),
                    fill="black", outline="black"
                )

    # def undo_overlay(self):

    #     self.binary_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    #     self.show_image(self.binary_image)

    def undo_overlay(self):
        if selected_boxes:
            selected_boxes.clear()
            self.show_image(self.binary_image)
            messagebox.showinfo("Undo", "Selection boxes cleared.")
            return

        if len(self.undo_stack) > 1:
            # self.undo_stack.pop() # Pop current state
            # self.binary_image = self.undo_stack[-1] # Restore previous state
            self.binary_image = self.undo_stack.pop()
            self.show_image(self.binary_image)
            messagebox.showinfo("Undo", "Last annotation undone.")
        else:
            messagebox.showinfo("Undo", "No more actions to undo.")

    def erase_selected_region(self):
        if not selected_boxes:
            messagebox.showinfo("Erase", "No region selected.")
            return

        self.undo_stack.append(self.binary_image.copy())
        for (x1, y1, x2, y2) in selected_boxes:
            cv2.rectangle(self.binary_image, (x1, y1), (x2, y2), color=255, thickness=-1)

        selected_boxes.clear()
        self.show_image(self.binary_image)
        messagebox.showinfo("Erase", "Selected region(s) erased.")

    def enter_move_mode(self):
        if len(selected_boxes) != 1:
            messagebox.showinfo("Move", "Please select exactly one region to move.")
            return
        self.move_mode = True
        self.move_start_box = selected_boxes[0]
        self.root.config(cursor="fleur")
        messagebox.showinfo("Move", "Click the destination to move the selection.")

    def move_region(self, box, new_point):
        self.undo_stack.append(self.binary_image.copy())
        self.root.config(cursor="cross")

        x1, y1, x2, y2 = box
        roi = self.binary_image[y1:y2, x1:x2].copy()
        w, h = x2 - x1, y2 - y1

        cv2.rectangle(self.binary_image, (x1, y1), (x2, y2), color=255, thickness=-1)

        new_x1 = new_point[0]
        new_y1 = new_point[1]
        new_x2 = new_x1 + w
        new_y2 = new_y1 + h

        if new_x2 > self.binary_image.shape[1] or new_y2 > self.binary_image.shape[0]:
            messagebox.showwarning("Move", "Move exceeds image bounds. Restoring original.")
            self.undo_stack.pop() # Cancel the move
            return

        self.binary_image[new_y1:new_y2, new_x1:new_x2] = roi
        self.show_image(self.binary_image)

    def open_region_threshold_editor(self):
        if not selected_boxes:
            messagebox.showinfo("Threshold", "Please select a region first.")
            return
        RegionThresholdEditor(self)

    def save_braille_output(self):
        if self.binary_image is None:
            messagebox.showerror("Save Error", "No image to save.")
            return
        output = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2RGB)
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if save_path:
            Image.fromarray(output).save(save_path, dpi=(300, 300))
            messagebox.showinfo("Success", f"Saved to: {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrailleOCRApp(root)
    root.mainloop()