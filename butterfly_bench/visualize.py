from PIL import ImageDraw


def draw_prediction(image, record, truth=None):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for boxes, color in [(truth or [], "lime"), (record["boxes"] or [], "red")]:
        for box in boxes:
            x, y = box["left"] * image.width, box["top"] * image.height
            draw.rectangle([x, y, x + box["width"] * image.width, y + box["height"] * image.height],
                           outline=color, width=max(2, image.width // 350))
    if record["error"]:
        draw.text((10, 10), "PROCESSING ERROR", fill="red", stroke_width=1)
    return image

