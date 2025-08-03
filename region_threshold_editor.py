import tkinter as tk
from PIL import Image, ImageTk
import cv2


class RegionThresholdEditor:
    def __init__(self, parent_app):
        self.parent = parent_app
        self.orig_image = parent_app.gray_image.copy()
        self.is_inverted = parent_app.is_inverted
        self.scale = parent_app.preview_scale
        self.a4_width = parent_app.a4_width
        self.a4_height = parent_app.a4_height

        self.local_boxes = []  # Only used inside the popup

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

        self.output_image = self.parent.binary_image.copy()  # Editable binary image
        self.update_preview()

    def update_preview(self, *_):
        # Start from original grayscale
        canvas_img = self.parent.embed_in_a4_canvas(self.orig_image.copy())

        if self.is_inverted:
            canvas_img = cv2.bitwise_not(canvas_img)

        for (x1, y1, x2, y2) in self.local_boxes:
            # Apply threshold only to ROI
            roi = canvas_img[y1:y2, x1:x2]
            _, roi_thresh = cv2.threshold(roi, int(self.slider.get()), 255, cv2.THRESH_BINARY)
            canvas_img[y1:y2, x1:x2] = roi_thresh

        self.output_image = canvas_img.copy()
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
        self.parent.binary_image = self.output_image.copy()
        self.parent.show_image(self.parent.binary_image)
        self.window.destroy()
