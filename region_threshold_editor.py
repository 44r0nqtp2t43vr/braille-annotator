import tkinter as tk
from PIL import Image, ImageTk
import cv2


class RegionThresholdEditor:
    def __init__(self, parent_app):
        self.parent = parent_app
        self.scale = parent_app.preview_scale
        self.a4_width = parent_app.a4_width
        self.a4_height = parent_app.a4_height
        self.is_inverted = parent_app.is_inverted

        # Embed grayscale image once into A4 canvas
        self.embedded_gray = parent_app.embed_in_a4_canvas(parent_app.gray_image.copy())
        if self.is_inverted:
            self.embedded_gray = cv2.bitwise_not(self.embedded_gray)

        # Start with a copy of the current binary image
        self.output_image = parent_app.binary_image.copy()

        self.local_boxes = []

        # Setup popup window
        self.window = tk.Toplevel(parent_app.root)
        self.window.title("Set Region Threshold")

        self.canvas_width = int(self.a4_width * self.scale)
        self.canvas_height = int(self.a4_height * self.scale)

        self.canvas = tk.Canvas(
            self.window,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="black",
            cursor="cross"
        )
        self.canvas.pack()

        self.slider = tk.Scale(
            self.window,
            from_=0,
            to=255,
            orient="horizontal",
            label="Threshold",
            command=self.update_preview
        )
        self.slider.set(128)
        self.slider.pack(fill="x")

        btn_frame = tk.Frame(self.window)
        btn_frame.pack()
        tk.Button(btn_frame, text="Apply Changes", command=self.apply_changes).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Cancel", command=self.window.destroy).pack(side="left", padx=10)

        self.drawing = False
        self.ix = self.iy = -1
        self.tk_img = None

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.update_preview()

    def update_preview(self, *_):
        # Start from the current output image
        canvas_img = self.output_image.copy()

        # Apply threshold only to selected regions
        for (x1, y1, x2, y2) in self.local_boxes:
            roi = self.embedded_gray[y1:y2, x1:x2].copy()
            _, roi_thresh = cv2.threshold(roi, int(self.slider.get()), 255, cv2.THRESH_BINARY)
            canvas_img[y1:y2, x1:x2] = roi_thresh

        self.preview_image = canvas_img
        self.show_image()

    def show_image(self):
        img = cv2.resize(self.preview_image, (self.canvas_width, self.canvas_height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

        for (x1, y1, x2, y2) in self.local_boxes:
            self.canvas.create_rectangle(
                x1 * self.scale, y1 * self.scale,
                x2 * self.scale, y2 * self.scale,
                outline="lime", width=2
            )

    def on_mouse_down(self, event):
        self.drawing = True
        self.ix = int(self.canvas.canvasx(event.x) / self.scale)
        self.iy = int(self.canvas.canvasy(event.y) / self.scale)

    def on_mouse_up(self, event):
        if not self.drawing:
            return
        ex = int(self.canvas.canvasx(event.x) / self.scale)
        ey = int(self.canvas.canvasy(event.y) / self.scale)
        x1, y1 = min(self.ix, ex), min(self.iy, ey)
        x2, y2 = max(self.ix, ex), max(self.iy, ey)
        self.local_boxes.append((x1, y1, x2, y2))
        self.update_preview()
        self.drawing = False

    def apply_changes(self):
        # Apply only selected region thresholds to parent binary image
        for (x1, y1, x2, y2) in self.local_boxes:
            self.parent.binary_image[y1:y2, x1:x2] = self.preview_image[y1:y2, x1:x2]

        # Optionally save regions if needed in parent
        if hasattr(self.parent, "selected_boxes"):
            self.parent.selected_boxes.clear()
            self.parent.selected_boxes.extend(self.local_boxes)

        self.parent.show_image(self.parent.binary_image)
        self.window.destroy()
