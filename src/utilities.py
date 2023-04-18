import cv2
import numpy as np

def padded_resize(img, new_size):
    h, w, _ = img.shape
    new_w, new_h = new_size
    aspect_ratio = w / h

    # calculate new size with aspect ratio preserved
    if new_w / aspect_ratio <= new_h:
        resize_w = int(new_w)
        resize_h = int(new_w / aspect_ratio)
    else:
        resize_w = int(new_h * aspect_ratio)
        resize_h = int(new_h)

    # resize image using INTER_AREA interpolation
    resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

    # add padding to fill the new dimensions
    top_pad = (new_h - resize_h) // 2
    bottom_pad = new_h - resize_h - top_pad
    left_pad = (new_w - resize_w) // 2
    right_pad = new_w - resize_w - left_pad
    padded = cv2.copyMakeBorder(resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded



def put_Text(
    img,
    text,
    uv_top_left,
    bg_color=(255, 255, 255),
    text_color=(255, 255, 255),
    fontScale=0.6,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0,0,0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline and a background slightly larger than the text.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    # Calculate the size of the text box
    text_size = np.zeros((2,))
    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        text_size[0] = max(text_size[0], w)
        text_size[1] += h * line_spacing

    # Calculate the top-left and bottom-right coordinates of the background box
    bg_size = text_size + np.array([10, 10])  # add some padding around the text
    bg_top_left = uv_top_left - [5, 5]
    bg_bottom_right = bg_top_left + bg_size

    # Draw the background box
    cv2.rectangle(
        img=img,
        pt1=tuple(bg_top_left.astype(int)),
        pt2=tuple(bg_bottom_right.astype(int)),
        color=bg_color,
        thickness=-1,
    )

    # Draw the text
    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=text_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]
