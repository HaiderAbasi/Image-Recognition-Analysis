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

putText_endcol = None
putText_row = None

def put_Text(
    img,
    text,
    uv_top_left,
    bg_color=(255, 255, 255),
    text_color="mixed",
    fontScale=0.6,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0,0,0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline and a background slightly larger than the text.
    """
    global putText_endcol,putText_row
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

    putText_endcol = bg_bottom_right.astype(int)[0]
    putText_row = int((bg_top_left[1] + bg_bottom_right[1])/2)
    # Draw the background box
    cv2.rectangle(
        img=img,
        pt1=tuple(bg_top_left.astype(int)),
        pt2=tuple(bg_bottom_right.astype(int)),
        color=bg_color,
        thickness=-1,
    )
    # # Draw the background box
    # cv2.rectangle(
    #     img=img,
    #     pt1=tuple(bg_top_left.astype(int)),
    #     pt2=tuple(bg_bottom_right.astype(int)),
    #     color=(52,78,247),
    #     thickness=2,
    # )
    
    color_list = [
        (255, 0, 255),
        (0, 204, 0),
        (255, 255, 51),
        (0, 255, 255)
        ]

    # Draw the text
    for i, line in enumerate(text.splitlines()):
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
                thickness=thickness ,
                lineType=cv2.LINE_AA,
            )
        
        txt_color = text_color
        if text_color == "mixed":
            txt_color = color_list[i]
        
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=txt_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def show_clr(cv_img,dominant_clr,clr):

    color_loc = (putText_endcol + 40,putText_row + 5)
    # Add the number of duplicates on the top right corner of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    r, g, b = dominant_clr
    text_size, _ = cv2.getTextSize(clr, font, font_scale, font_thickness)

    rect_x = color_loc[0] - 20
    rect_y = color_loc[1] - 20
    rect_w = text_size[0] + 30
    cv2.rectangle(cv_img,(rect_x,rect_y), (rect_x+rect_w, color_loc[1]+10), (255,255,255), -1)
    cv2.putText(cv_img, clr,color_loc,font,font_scale,(0,0,0),2)

    cv2.rectangle(cv_img, (color_loc[0]-40,color_loc[1]-20), (color_loc[0]-10,color_loc[1]+10), (b,g,r), -1)
    cv2.rectangle(cv_img, (color_loc[0]-40,color_loc[1]-20), (color_loc[0]-10,color_loc[1]+10), (0,0,0), 3)
