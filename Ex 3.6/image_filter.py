from PIL import Image, ImageTk
import tkinter as tk
import cv2
import argparse
from pathlib import Path

from handlers import RootHandler, ImageHandler


def _on_close(root):
    """
    > The function `_on_close` takes a single argument `root` and prints a message to the console when
    the window is closed

    :param root: the root window (Tk object)
    """
    print("[INFO] closing...")
    root.destroy()


if __name__ == "__main__":
    # Parsing the arguments that are passed in the command line.
    parser = argparse.ArgumentParser(description="Arguments for image filters")
    parser.add_argument(
        "-i", "--img_path", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./snapshots",
        help="Path to save the filtered images",
    )
    args = parser.parse_args()

    # Converting the string path to a Path object.
    out_path = Path(args.out_path)

    # This is creating a window with the title "Image Filter"
    root = tk.Tk()
    root.title("Image Filter")
    root.wm_protocol("WM_DELET_WINDOW", lambda: _on_close(root))
    # root.geometry("500x500")t

    # Reading the image from the path that is passed in the command line.
    img = cv2.imread(args.img_path)

    # Converting the image from BGR to RGB.
    img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img2 = ImageTk.PhotoImage(img2)

    # Creating two labels, one for the original image and one for the filtered image.
    orig = tk.Label(root, image=img2)
    orig.grid(row=0, column=0)
    
    panel = tk.Label(root, image=img2)
    panel.grid(row=0, column=1)

    # Creating a button that calls the `save_img` function in the `ImageHandler` class.
    btn = tk.Button(root, text="Save", bd=5, command=lambda: img_handler.save_img())
    btn.grid(row=1, column=1)

    # Creating an instance of the `ImageHandler` class and passing the original image, the filtered
    # image, and the output path.
    img_handler = ImageHandler(img, img, out_path)
    root_handler = RootHandler(panel)

    root_handler.bind_root(root, img_handler, init=True)
    root.bind("<Escape>", lambda e: _on_close(root))
    root.mainloop()
