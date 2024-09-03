import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2 
from typing import Dict, List, Optional

def append_text_to_image(image: np.ndarray, lines: List[str]):
    r"""Appends text left to an image of size (height, width, channels).
    The returned image has white text on a black background.
    Args:
        image: the image to put text
        text: a string to display
    Returns:
        A new image with text inserted left to the input image
    See also:
        habitat.utils.visualization.utils
    """
    # h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    final = np.concatenate((blank_image, image), axis=1)
    return final

def put_text_on_image(image: np.ndarray, lines: List[str]):
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()

    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return image

def format_color(color):
    if type(color) == str:
        color_str = color
    elif type(color) == np.ndarray and len(color) == 3:
        color = color.astype(np.int32)
        color_str = f'rgb({color[0]},{color[1]},{color[2]})'
    else:
        color = color.astype(np.int32)
        color_str = [f'rgb({color[k, 0]},{color[k, 1]},{color[k, 2]})' for k in range(color.shape[0])]
    return color_str

def plot_pcd(name, points, color, size=3):
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", 
                        marker = dict(color=format_color(color), size=size), name=name)

def plot_pcd_with_score(name, points, action_score, size=3):
    assert type(action_score) == np.ndarray
    action_score = action_score.reshape(-1)
    action_score = (action_score - action_score.min())/(action_score.max()-action_score.min()+1e-7)
    object_color_id = (action_score*255).astype(np.int32)
    object_color = []
    for cid in object_color_id:
        try:
            object_color.append(plt.get_cmap('plasma').colors[cid])
        except:
            print(f'cid={cid} gives an error. Will use 0 instead.')
            object_color.append(plt.get_cmap('plasma').colors[0])
    object_color = np.array(object_color)*255
    return plot_pcd(name, points, object_color, size)

def plot_action(name, start, direction, color='red', size=3):
    # direction = direction*0.02*3  # action scale=0.02. steps=10
    if start is None: x, y, z = 0, 0, 0
    else: x, y, z = start[0], start[1], start[2]
    try:
        u, v, w = direction[0], direction[1], direction[2]
    except:
        u, v, w = direction[0], direction[0], direction[0]
    return go.Scatter3d(x=[x, x + u], y=[y, y + v], z=[z, z + w], mode='lines',
                        line=dict(color=format_color(color), width=10), name=name)

def plot_actions(name, starts, directions, color='grey', size=1, legendgroup=None):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = (directions == 0.0).all(axis=-1)
    nz_start = starts[~nonzero_flows]
    nz_flows = directions[~nonzero_flows]

    nz_end = nz_start + nz_flows

    # Evil hacky line segments in 3D.
    for i in range(len(nz_start)):
        x_lines.append(nz_start[i][0])
        y_lines.append(nz_start[i][1])
        z_lines.append(nz_start[i][2])
        x_lines.append(nz_end[i][0])
        y_lines.append(nz_end[i][1])
        z_lines.append(nz_end[i][2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        line=dict(color=color, width=3),
        name=name,
        legendgroup=legendgroup,
    )

    return lines_trace