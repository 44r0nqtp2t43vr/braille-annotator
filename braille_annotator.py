import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if needed

selected_boxes = []
drawing = False
ix, iy = -1, -1

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
        
        self.canvas = tk.Canvas(root, cursor="cross", bg="gray20")
        self.canvas.pack(fill="both", expand=True)

        self.slider = tk.Scale(root, from_=0, to=255, orient="horizontal",
                               label="Threshold", command=self.update_threshold)
        self.slider.set(128)
        self.slider.pack(fill="x", padx=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.save_button = tk.Button(btn_frame, text="Save Braille Output", command=self.save_braille_output)
        self.save_button.pack(side="left", padx=5)

        self.preview_button = tk.Button(btn_frame, text="Preview Braille Text", command=self.preview_braille_text)
        self.preview_button.pack(side="left", padx=5)

        self.region_thresh_button = tk.Button(btn_frame, text="Set Region Threshold", command=self.set_region_threshold)
        self.region_thresh_button.pack(side="left", padx=5)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.orig_image = None
        self.tk_img = None
        self.scale = 1.0
        self.load_image()

    def load_image(self):
        selected_boxes.clear()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not path:
            self.root.quit()

        self.orig_image = cv2.imread(path)
        self.gray_image = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2GRAY)
        self.binary_image = None
        self.update_threshold(self.slider.get())

    def update_threshold(self, value):
        if not hasattr(self, 'gray_image') or self.gray_image is None:
            return
        thresh_val = int(value)
        _, thresh = cv2.threshold(self.gray_image, thresh_val, 255, cv2.THRESH_BINARY)
        self.binary_image = thresh
        self.show_image(thresh)

    def show_image(self, image):
        # Always draw from full resolution but show scaled preview (fit-to-screen)
        h, w = image.shape
        screen_h = self.root.winfo_screenheight() - 200
        screen_w = self.root.winfo_screenwidth() - 100
        self.scale = min(screen_w / w, screen_h / h, 1.0)

        disp_img = cv2.resize(image, (int(w * self.scale), int(h * self.scale)))
        rgb = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image((screen_w - disp_img.shape[1]) // 2, (screen_h - disp_img.shape[0]) // 2,
                                 image=self.tk_img, anchor="nw", tags="img")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def set_region_threshold(self):
        if not selected_boxes:
            messagebox.showinfo("Info", "No region selected. Draw a box first.")
            return
        # Use the last selected box
        x1, y1, x2, y2 = selected_boxes[-1]
        thresh_val = tk.simpledialog.askinteger("Region Threshold", "Enter threshold value (0-255):", minvalue=0, maxvalue=255)
        if thresh_val is None:
            return
        # Apply threshold only to the selected region
        region = self.gray_image[y1:y2, x1:x2]
        _, region_thresh = cv2.threshold(region, thresh_val, 255, cv2.THRESH_BINARY)
        # Update the binary image only in the region
        self.binary_image[y1:y2, x1:x2] = region_thresh
        self.show_image(self.binary_image)

    def on_mouse_down(self, event):
        global ix, iy, drawing
        drawing = True
        # Calculate offset due to centering
        screen_h = self.root.winfo_screenheight() - 200
        screen_w = self.root.winfo_screenwidth() - 100
        h, w = self.binary_image.shape
        disp_w, disp_h = int(w * self.scale), int(h * self.scale)
        offset_x = (screen_w - disp_w) // 2
        offset_y = (screen_h - disp_h) // 2
        ix = self.canvas.canvasx(event.x) - offset_x
        iy = self.canvas.canvasy(event.y) - offset_y

    def on_mouse_up(self, event):
        global drawing
        if drawing:
            drawing = False
            screen_h = self.root.winfo_screenheight() - 200
            screen_w = self.root.winfo_screenwidth() - 100
            h, w = self.binary_image.shape
            disp_w, disp_h = int(w * self.scale), int(h * self.scale)
            offset_x = (screen_w - disp_w) // 2
            offset_y = (screen_h - disp_h) // 2
            ex = self.canvas.canvasx(event.x) - offset_x
            ey = self.canvas.canvasy(event.y) - offset_y
            # Only allow coordinates inside the displayed image
            ex = min(max(ex, 0), disp_w)
            ey = min(max(ey, 0), disp_h)
            ix_clamped = min(max(ix, 0), disp_w)
            iy_clamped = min(max(iy, 0), disp_h)
            x1, y1 = int(ix_clamped / self.scale), int(iy_clamped / self.scale)
            x2, y2 = int(ex / self.scale), int(ey / self.scale)
            print(f"Raw coords: ix={ix}, iy={iy}, ex={ex}, ey={ey}")
            print(f"Image coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            selected_boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            print("Selected boxes:", selected_boxes)
            self.canvas.create_rectangle(ix + offset_x, iy + offset_y, ex + offset_x, ey + offset_y, outline="lime", width=2)

    def extract_text_and_braille(self):
        braille_output = []
        for (x1, y1, x2, y2) in selected_boxes:
            # Ensure coordinates are within image bounds and region is valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.binary_image.shape[1], x2), min(self.binary_image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid box: {(x1, y1, x2, y2)}")
                continue
            roi = self.binary_image[y1:y2, x1:x2]
            print("ROI shape:", roi.shape, "ROI dtype:", roi.dtype)
            if roi.size == 0:
                continue  # Skip invalid regions
            text = pytesseract.image_to_string(roi, config="--psm 6").strip()
            print("OCR text:", repr(text))
            braille = '\n'.join(
                ''.join([char_to_braille(c) for c in line])
                for line in text.splitlines()
            )
            braille_output.append((text, braille))
        return braille_output

    def preview_braille_text(self):
        if not selected_boxes:
            messagebox.showinfo("Info", "No boxes selected.")
            return
        text_pairs = self.extract_text_and_braille()
        print("Extracted text and braille:", text_pairs)
        preview = "\n\n".join([
            f"Box: {box}\nText: {t}\nBraille: {b}"
            for box, (t, b) in zip(selected_boxes, text_pairs)
        ])

        preview_window = tk.Toplevel(self.root)
        preview_window.title("Braille Preview")
        preview_window.geometry("500x300")

        btn_frame = tk.Frame(preview_window)
        btn_frame.pack(fill="x", pady=5, side="top")

        text_box = tk.Text(preview_window, wrap="word")
        text_box.insert("1.0", preview)
        text_box.pack(expand=True, fill="both")

        def replace_with_braille():
            edited = text_box.get("1.0", "end").strip()
            blocks = [block.strip() for block in edited.split("\n\n") if block.strip()]
            new_pairs = []
            for block in blocks:
                lines = block.splitlines()
                box_line = next((line for line in lines if line.startswith("Box: ")), None)
                text_line = next((line for line in lines if line.startswith("Text: ")), None)
                braille_start = None
                for idx, line in enumerate(lines):
                    if line.startswith("Braille: "):
                        braille_start = idx
                        break
                if box_line and text_line and braille_start is not None:
                    box_str = box_line[len("Box: "):].strip()
                    box = tuple(map(int, box_str.strip("()").split(",")))
                    text = text_line[len("Text: "):]
                    # Get all lines after "Braille: ", including the first line after the prefix
                    braille_lines = [lines[braille_start][len("Braille: "):]] + lines[braille_start+1:]
                    braille_lines = [line.strip() for line in braille_lines]
                    braille = "\n".join(braille_lines)
                    new_pairs.append((box, (text, braille)))
            # Overlay using the edited Braille text and the correct boxes
            self.overlay_braille_on_image(new_pairs)
            # Optionally, update selected_boxes to match the new set:
            global selected_boxes
            selected_boxes = [box for box, _ in new_pairs]
            preview_window.destroy()

        replace_btn = tk.Button(btn_frame, text="Replace Box with Braille", command=replace_with_braille)
        replace_btn.pack(pady=5)

    def overlay_braille_on_image(self, text_pairs):
        img = self.binary_image.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)

        font_path = "DejaVuSans-Bold.ttf"
        try:
            base_font = ImageFont.truetype(font_path, 40)
        except Exception as e:
            print("Font load error:", e)
            base_font = ImageFont.load_default()

        draw = ImageDraw.Draw(pil_img)

        def wrap_braille_lines(braille, font, max_width):
            # Split the braille text into lines using \n, then wrap each line by width
            user_lines = braille.split('\n')
            lines = []
            for user_line in user_lines:
                words = [w for w in user_line.split(' ') if w.strip() != '']
                current = ""
                for word in words:
                    test = (current + ' ' + word).strip() if current else word
                    bbox = draw.textbbox((0, 0), test, font=font)
                    w = bbox[2] - bbox[0]
                    if w > max_width and current:
                        lines.append(current)
                        current = word
                    else:
                        current = test
                if current or not words:  # preserve empty lines
                    lines.append(current)
            return lines

        for (x1, y1, x2, y2), (text, braille) in text_pairs:
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
            box_w = x2 - x1
            box_h = y2 - y1

            if braille:
                best_font_size = 10
                best_lines = [braille]
                for size in range(10, 200):
                    try:
                        test_font = ImageFont.truetype(font_path, size)
                    except Exception:
                        test_font = base_font
                    lines = wrap_braille_lines(braille, test_font, box_w)
                    line_heights = [draw.textbbox((0, 0), line, font=test_font)[3] - draw.textbbox((0, 0), line, font=test_font)[1] for line in lines]
                    total_height = sum(line_heights) + (len(lines) - 1) * 4
                    if total_height > box_h:
                        break
                    if any(draw.textbbox((0, 0), line, font=test_font)[2] - draw.textbbox((0, 0), line, font=test_font)[0] > box_w for line in lines):
                        break
                    best_font_size = size
                    best_lines = lines

                try:
                    font = ImageFont.truetype(font_path, best_font_size)
                except Exception:
                    font = base_font

                line_heights = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in best_lines]
                total_height = sum(line_heights) + (len(best_lines) - 1) * 4
                y = y1 + (box_h - total_height) // 2

                for line in best_lines:
                    if line:
                        bbox = draw.textbbox((0, 0), line, font=font)
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        x = x1 + (box_w - w) // 2
                        draw.text((x, y), line, font=font, fill=(0, 0, 0))
                    else:
                        # For empty lines, estimate height using a typical character
                        h = draw.textbbox((0, 0), "A", font=font)[3] - draw.textbbox((0, 0), "A", font=font)[1]
                    y += h + 4

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        self.show_image(img_bgr)
        self.binary_image = img_bgr

    def save_braille_output(self):
        # Convert binary image to BGR for colored overlay
        output = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
        for (x1, y1, x2, y2) in selected_boxes:
            roi = self.binary_image[y1:y2, x1:x2]
            text = pytesseract.image_to_string(roi, config="--psm 6").strip()
            for idx, ch in enumerate(text):
                braille_char = char_to_braille(ch)
                # Draw Braille text in color (e.g., green) for visibility
                cv2.putText(output, braille_char, (x1 + idx * 20, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG", "*.png")])
        if save_path:
            cv2.imwrite(save_path, output)
            messagebox.showinfo("Success", f"Saved to: {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrailleOCRApp(root)
    root.mainloop()
