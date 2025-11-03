import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

# === Settings ===
video_path = "sample.mp4"
max_speed = 200.0
movement_threshold = 0.3
smoothing = 0.8
buffer_len = 10

# === Default toggles ===
show_pose = True
draw_on_video = False
show_bike = False
show_motion_info = True
detection_enabled = True
paused = True
speed_m_s = 0.0
total_dist_m = 0.0

# === Animation controls for pose subframe ===
alpha_pose = 0.0
target_alpha_pose = 1.0 if show_pose else 0.0
fade_speed = 0.15  # speed of pose subframe fade

# === Shared overlay controls (Help / About) ===
overlay_open = False
overlay_mode = None            # "help" or "about"
overlay_alpha = 0.0
overlay_target_alpha = 0.0
overlay_fade_speed = 0.18      # controls fade duration ~300ms at 60fps
detection_enabled_before_overlay = True
overlay_box = None             # (x1,y1,x2,y2) current overlay content box (for click detection)
overlay_coords = None          # store (x1,y1,x2,y2,scale) for click math

# === Hover variables for close icon and menu ===
hover_close = False
hover_alpha = 0.0

menu_hover_index = -1
menu_hover_alphas = []  # will initialize after menu_items length

# === Menu settings ===
menu_open = True
menu_anim_x = 1.0
menu_target = 1.0
menu_speed = 0.12
menu_width = 280
menu_items = [
    ("Pose", "show_pose"),
    ("Bike", "show_bike"),
    ("Motion Info", "show_motion_info"),
    ("Skeleton", "draw_on_video"),
    ("Detection", "detection_enabled"),
    ("Help", "help"),
    ("About", "about"),
    ("Reset Dist", "reset"),
    ("Quit", "quit")
]
menu_rects = []

# initialize menu hover alphas
menu_hover_alphas = [0.0 for _ in menu_items]

# === Real-world Estimation Settings ===
camera_fov_deg = 60.0
subject_distance_m = 2.0

# === Utility functions ===

def pixels_to_meters(pixel_distance, image_width, fov_deg, distance_m):
    fov_rad = np.deg2rad(fov_deg)
    meters_per_pixel = (2 * distance_m * np.tan(fov_rad / 2)) / image_width
    return 10.0 * pixel_distance * meters_per_pixel

def overlay_image_alpha(background, overlay, x, y, overlay_scale=1.0):
    overlay = cv2.resize(overlay, None, fx=overlay_scale, fy=overlay_scale, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]
    if x >= bg_w or y >= bg_h:
        return background
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    overlay_roi = overlay[0:(y2 - y1), 0:(x2 - x1)]
    if overlay_roi.shape[2] == 4:
        alpha = overlay_roi[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]
        color = overlay_roi[:, :, :3]
    else:
        alpha = np.ones((overlay_roi.shape[0], overlay_roi.shape[1], 1), dtype=float)
        color = overlay_roi
    background[y1:y2, x1:x2] = (alpha * color + (1 - alpha) * background[y1:y2, x1:x2]).astype(np.uint8)
    return background

# Helper: draw rounded rectangle into mask (filled)
def rounded_rect_mask(h, w, x1, y1, x2, y2, r):
    mask = np.zeros((h, w), dtype=np.uint8)
    # central rect
    cv2.rectangle(mask, (x1 + r, y1), (x2 - r, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1 + r), (x2, y2 - r), 255, -1)
    # four circles
    cv2.circle(mask, (x1 + r, y1 + r), r, 255, -1)
    cv2.circle(mask, (x2 - r, y1 + r), r, 255, -1)
    cv2.circle(mask, (x1 + r, y2 - r), r, 255, -1)
    cv2.circle(mask, (x2 - r, y2 - r), r, 255, -1)
    return mask

def draw_menu(frame):
    """Draws animated semi-transparent menu (slides from right) with hover effect."""
    global menu_rects, menu_anim_x, menu_hover_alphas, menu_hover_index

    overlay = frame.copy()
    alpha = 0.45
    spacing = 50
    w = menu_width
    h = 40
    margin = 18

    # animation X
    current_x = frame.shape[1] - int((menu_anim_x) * (w + margin))
    menu_anim_x += (menu_target - menu_anim_x) * menu_speed
    menu_rects.clear()

    y = margin
    total_height = spacing * len(menu_items)
    x1, y1 = current_x - 10, y - 10
    x2, y2 = current_x + w + 10, y + total_height + 8

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (24, 24, 28), -1)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # animate and draw each menu item with subtle hover brighten
    for i, (label, key) in enumerate(menu_items):
        rect_y1 = y + i * spacing
        rect_y2 = rect_y1 + h
        rect = (current_x, rect_y1, current_x + w, rect_y2)
        menu_rects.append((rect, key))

        # update hover alpha per item
        target = 1.0 if i == menu_hover_index else 0.0
        # smoothing factor for hover animation
        menu_hover_alphas[i] += (target - menu_hover_alphas[i]) * 0.22
        ha = float(np.clip(menu_hover_alphas[i], 0.0, 1.0))

        active = globals().get(key, False) if key not in ["help", "about", "reset", "quit"] else False

        if key == "help":
            base_color = np.array([90, 150, 230])
        elif key == "about":
            base_color = np.array([150, 120, 220])
        elif key == "reset":
            base_color = np.array([200, 150, 40])
        elif key == "quit":
            base_color = np.array([150, 50, 120])
        else:
            base_color = np.array([60, 60, 110]) if not active else np.array([0, 160, 80])

        # brighten a bit on hover
        color = tuple(np.clip(base_color + (40 * ha), 0, 255).astype(np.uint8).tolist())

        # slightly scale rect when hovered
        expand = int(4 * ha)
        cv2.rectangle(frame, (current_x - expand, rect_y1 - expand), (current_x + w + expand, rect_y2 + expand), color, -1)
        cv2.putText(frame, label, (current_x + 15, rect_y1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

def open_overlay(mode):
    """Open shared overlay (help or about) and pause detection."""
    global overlay_open, overlay_mode, overlay_target_alpha, detection_enabled_before_overlay, detection_enabled
    if not overlay_open:
        detection_enabled_before_overlay = detection_enabled
        detection_enabled = False
        overlay_open = True
        overlay_mode = mode
        overlay_target_alpha = 1.0
    else:
        # switch mode while open
        overlay_mode = mode
        overlay_target_alpha = 1.0

def close_overlay():
    """Start fade-out for overlay; detection will be restored when fade finishes."""
    global overlay_target_alpha
    overlay_target_alpha = 0.0

def mouse_callback(event, mx, my, flags, param):
    """Handle mouse clicks and move for menu and overlay (hover + click)."""
    global show_pose, show_bike, show_motion_info, draw_on_video, detection_enabled, total_dist_m, target_alpha_pose
    global overlay_open, overlay_box, overlay_coords, hover_close, menu_hover_index, menu_rects

    # Mouse MOVE: update hover states for menu buttons and overlay close icon
    if event == cv2.EVENT_MOUSEMOVE:
        # update menu hover
        menu_hover_index = -1
        for idx, (rect, key) in enumerate(menu_rects):
            x1, y1, x2, y2 = rect
            if x1 <= mx <= x2 and y1 <= my <= y2:
                menu_hover_index = idx
                break

        # update hover for close icon if overlay visible and we have overlay_coords
        if overlay_open and overlay_coords is not None:
            ox1, oy1, ox2, oy2, scale = overlay_coords
            # compute close base using same formula as draw_overlay
            close_base = int(22 * scale)
            cx = ox2 - close_base - 15
            cy = oy1 + close_base + 5
            # close hit area
            if (cx - close_base//2 <= mx <= cx + close_base//2 and
                cy - close_base//2 <= my <= cy + close_base//2):
                hover_close = True
            else:
                hover_close = False
        return  # on mouse move we don't need further click handling

    # If overlay is open: clicking outside the overlay box closes it. Clicking inside does nothing except close if ×
    # clicked.
    if event == cv2.EVENT_LBUTTONDOWN and overlay_open:
        if overlay_box is not None:
            x1, y1, x2, y2 = overlay_box
            # check if clicked on × first (we need overlay_coords to compute exact area)
            if overlay_coords is not None:
                ox1, oy1, ox2, oy2, scale = overlay_coords
                close_base = int(22 * scale)
                cx = ox2 - close_base - 15
                cy = oy1 + close_base + 5
                if (cx - close_base//2 <= mx <= cx + close_base//2 and
                    cy - close_base//2 <= my <= cy + close_base//2):
                    # close via ×
                    close_overlay()
                    return

            # click outside -> close
            if not (x1 <= mx <= x2 and y1 <= my <= y2):
                close_overlay()
                return
            else:
                # clicked inside (but not ×): do nothing (allow reading)
                return

    # otherwise, handle menu clicks
    if event == cv2.EVENT_LBUTTONDOWN and menu_open:
        for (rect, key) in menu_rects:
            x1, y1, x2, y2 = rect
            if x1 <= mx <= x2 and y1 <= my <= y2:
                if key == "show_pose":
                    show_pose = not show_pose
                    target_alpha_pose = 1.0 if show_pose else 0.0
                elif key == "show_bike":
                    show_bike = not show_bike
                elif key == "show_motion_info":
                    show_motion_info = not show_motion_info
                elif key == "draw_on_video":
                    draw_on_video = not draw_on_video
                elif key == "detection_enabled":
                    detection_enabled = not detection_enabled
                elif key == "help":
                    if not overlay_open or overlay_mode != "help":
                        open_overlay("help")
                    else:
                        close_overlay()
                elif key == "about":
                    if not overlay_open or overlay_mode != "about":
                        open_overlay("about")
                    else:
                        close_overlay()
                elif key == "reset":
                    total_dist_m = 0.0
                    print("Distance reset")
                elif key == "quit":
                    print("Quit requested.")
                    exit(0)

def draw_skeleton(frame, keypoints, pairs, color=(0, 255, 0)):
    if keypoints is None or len(keypoints) == 0:
        return frame
    for (a, b) in pairs:
        if a < len(keypoints) and b < len(keypoints):
            xa, ya = keypoints[a]
            xb, yb = keypoints[b]
            if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)
    return frame

# Unified overlay drawing (rounded box + soft shadow) that displays Help or About based on overlay_mode
def draw_overlay(frame):
    """Draw shared fade background + selected overlay content (rounded box + shadow)."""
    global overlay_alpha, overlay_target_alpha, overlay_open, overlay_mode, overlay_box, detection_enabled, detection_enabled_before_overlay
    global overlay_coords, hover_alpha, hover_close

    # Smooth fade
    overlay_alpha += (overlay_target_alpha - overlay_alpha) * overlay_fade_speed
    overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))

    # Close finished (fade-out completed)
    if overlay_target_alpha == 0.0 and overlay_alpha < 0.005 and overlay_open:
        overlay_open = False
        # restore detection
        detection_enabled = detection_enabled_before_overlay
        overlay_mode = None
        overlay_box = None
        overlay_coords = None
        hover_close = False
        return frame

    if overlay_alpha < 0.01:
        return frame

    h, w = frame.shape[:2]
    scale = min(w, h) / 1080.0
    title_scale = 0.95 * scale
    text_scale = 0.65 * scale
    spacing = int(30 * scale)

    box_w = int(w * 0.72)
    box_h = int(h * 0.56)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    r = int(min(box_w, box_h) * 0.03)  # corner radius

    # produce shadow by drawing a larger rounded mask, blurring and compositing
    shadow = np.zeros_like(frame, dtype=np.uint8)
    sx1, sy1, sx2, sy2 = x1 - 12, y1 - 12, x2 + 12, y2 + 12
    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(w, sx2), min(h, sy2)
    mask_shadow = rounded_rect_mask(h, w, sx1, sy1, sx2, sy2, max(6, r+6))
    shadow[:,:,0] = mask_shadow
    shadow[:,:,1] = mask_shadow
    shadow[:,:,2] = mask_shadow
    blurred = cv2.GaussianBlur(shadow, (21,21), 0)
    shadow_alpha = 0.45 * overlay_alpha
    frame = cv2.addWeighted(blurred, shadow_alpha, frame, 1 - shadow_alpha, 0)

    # draw the rounded rectangle background (slightly translucent)
    mask_box = rounded_rect_mask(h, w, x1, y1, x2, y2, r)
    box_color = np.array([18, 18, 20], dtype=np.uint8)
    box_img = np.zeros_like(frame, dtype=np.uint8)
    box_img[:] = box_color
    mask_3c = np.stack([mask_box]*3, axis=-1) // 255
    frame = np.where(mask_3c==1, cv2.addWeighted(box_img, 0.88 * overlay_alpha, frame, 1 - 0.88 * overlay_alpha, 0), frame)

    # border
    border_color = (220, 220, 220)
    cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), border_color, 1)

    # remember overlay box for click detection (inner content box)
    overlay_box = (x1+4, y1+4, x2-4, y2-4)
    # store coords and scale for mouse math (close icon pos)
    overlay_coords = (x1, y1, x2, y2, scale)

    # --- Close icon (×) with hover glow and size animation ---
    close_base = int(22 * scale)
    # compute animated hover alpha
    target_hover_alpha = 1.0 if hover_close else 0.0
    hover_alpha += (target_hover_alpha - hover_alpha) * overlay_fade_speed
    hover_alpha = float(np.clip(hover_alpha, 0.0, 1.0))
    # sizes
    close_size = int(close_base * (1.0 + 0.2 * hover_alpha))
    close_th = max(1, int(2 * scale * (1.0 + 0.3 * hover_alpha)))
    cx = x2 - close_base - 15
    cy = y1 + close_base + 5

    # Draw hover glow (soft white circle)
    if hover_alpha > 0.01:
        glow_radius = int(close_base * 2.0)
        glow = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(glow, (cx, cy), glow_radius, (255, 255, 255), -1)
        glow = cv2.GaussianBlur(glow, (51, 51), 0)
        frame = cv2.addWeighted(glow, 0.25 * hover_alpha, frame, 1 - 0.25 * hover_alpha, 0)

    # Draw × icon with slight brightening
    color_val = int(220 + 35 * hover_alpha)
    cv2.line(frame, (cx - close_size//2, cy - close_size//2),
             (cx + close_size//2, cy + close_size//2), (color_val, color_val, color_val), close_th)
    cv2.line(frame, (cx + close_size//2, cy - close_size//2),
             (cx - close_size//2, cy + close_size//2), (color_val, color_val, color_val), close_th)

    # Draw content
    title_x = x1 + int(30 * scale)
    title_y = y1 + int(60 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if overlay_mode == "help":
        cv2.putText(frame, "Help - How to use this app", (title_x, title_y),
                    font, title_scale, (245,245,245), 2)
        lines = [
            "Press 'm' - Show / Hide the menu (top-right).",
            "Click menu buttons to toggle features.",
            "Press 'p' - Toggle pose mini-view (bottom-right).",
            "Press 's' - Toggle skeleton overlay on main video.",
            "Press 'd' - Enable/disable detection (YOLO).",
            "Press 'h' - Open/close Help.",
            "Press 'a' - Open/close About.",
            "Detection pauses while this window is open.",
            "Click outside or on x to close overlay.",
        ]
        y_text = title_y + 40
        for line in lines:
            cv2.putText(frame, line, (title_x, y_text),
                        font, text_scale, (220,220,220), 1)
            y_text += spacing

    elif overlay_mode == "about":
        cv2.putText(frame, "About - Ali Abbas", (title_x, title_y),
                    font, title_scale, (245,245,245), 2)

        # About section
        y_text = title_y + int(40 * scale)
        cv2.putText(frame, "About", (title_x, y_text),
                    font, text_scale + 0.1, (230,230,230), 1)
        y_text += spacing + 5
        about_lines = [
            "PhD-qualified Software Engineer specializing in Computer Vision, GIS, and AI.",
            "20+ years of experience across embedded systems, UAV navigation, 3D mapping, and full-stack development."
        ]
        for line in about_lines:
            cv2.putText(frame, line, (title_x, y_text),
                        font, text_scale, (220,220,220), 1)
            y_text += spacing

        # Contact section
        y_text += int(10 * scale)
        cv2.putText(frame, "Contact", (title_x, y_text),
                    font, text_scale + 0.1, (230,230,230), 1)
        y_text += spacing + 5
        contact_lines = [
            "Email: aabbas7@gmail.com",
            "GitHub: github.com/aabbas77-web",
            "LinkedIn: linkedin.com/in/ali-abbas-45799710b",
            "",
            "Click outside this box or press 'a' to close."
        ]
        for line in contact_lines:
            cv2.putText(frame, line, (title_x, y_text),
                        font, text_scale, (210,210,210), 1)
            y_text += spacing

    return frame

def show_video_frame(frame, pose_subframe=None):
    """Draw overlays, menu, pose subframe and shared overlay, then show."""
    global paused, total_dist_m, speed_m_s, alpha_pose, target_alpha_pose

    # Smooth alpha transition for pose subframe
    alpha_pose += (target_alpha_pose - alpha_pose) * fade_speed
    alpha_pose = float(np.clip(alpha_pose, 0.0, 1.0))

    h, w, _ = frame.shape
    if show_bike and bike_img is not None:
        bike_h, bike_w = bike_img.shape[:2]
        scale = 0.3
        x_pos = int((w - bike_w * scale) / 2)
        y_pos = int(h - bike_h * scale)
        frame = overlay_image_alpha(frame, bike_img, x_pos, y_pos, overlay_scale=scale)

    color = (0, 0, 200) if paused else (0, 180, 20)
    status_text = "STOPPED" if paused else "Running"
    
    # Motion info text with soft glow
    if show_motion_info:
        glow = frame.copy()
        cv2.putText(glow, f"Distance: {total_dist_m:.1f} m | Speed: {speed_m_s:.2f} m/s",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 6)
        cv2.putText(glow, status_text, (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 6)
        frame = cv2.addWeighted(glow, 0.25, frame, 0.75, 0)

        cv2.putText(frame, f"Distance: {total_dist_m:.1f} m | Speed: {speed_m_s:.2f} m/s",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 23, 50), 2)
        cv2.putText(frame, status_text, (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Pose sub-view (fade)
    if alpha_pose > 0.01 and pose_subframe is not None:
        h_pose, w_pose = pose_subframe.shape[:2]
        scale = 0.25
        small_pose = cv2.resize(pose_subframe, (int(w_pose * scale), int(h_pose * scale)))
        ph, pw = small_pose.shape[:2]
        x_offset = w - pw - 14
        y_offset = h - ph - 14

        # translucent black background for subframe (blended by alpha_pose)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset - 6, y_offset - 6),
                      (x_offset + pw + 6, y_offset + ph + 6), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45 * alpha_pose, frame, 1 - 0.45 * alpha_pose, 0)

        # blend the small pose into ROI according to alpha_pose
        roi = frame[y_offset:y_offset + ph, x_offset:x_offset + pw]
        blended = cv2.addWeighted(small_pose, alpha_pose, roi, 1 - alpha_pose, 0)
        frame[y_offset:y_offset + ph, x_offset:x_offset + pw] = blended

        # border
        cv2.rectangle(frame, (x_offset - 2, y_offset - 2),
                      (x_offset + pw + 2, y_offset + ph + 2), (230, 230, 230), 1)

    # draw menu on top
    draw_menu(frame)

    # draw shared overlay (help/about) on top if active
    if overlay_open or overlay_alpha > 0.001 or overlay_target_alpha > 0.0:
        frame = draw_overlay(frame)

    cv2.imshow("Controlled Video", frame)

# === Load resources ===
# Load a pre-trained pose model
model = YOLO("yolov8n-pose.pt")  # or yolov8s-pose.pt, yolov8m-pose.pt

cap_cam = cv2.VideoCapture(0)
cap_video = cv2.VideoCapture(video_path)
if not cap_cam.isOpened():
    raise RuntimeError("Webcam not found!")
if not cap_video.isOpened():
    raise RuntimeError("Video file not found!")

cv2.namedWindow("Controlled Video", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Controlled Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Controlled Video", mouse_callback)
bike_img = cv2.imread("bike.png", cv2.IMREAD_UNCHANGED)

prev_keypoints = None
prev_time = time.time()
smoothed_speed = 0.0
speed_buffer = deque(maxlen=buffer_len)
frame_video = None

skeleton_pairs = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

print("System ready - press [m] to toggle the menu, 'h' for Help, 'a' for About.")

# === Main loop ===
while True:
    # Always read camera frame (we keep last video frame frozen when detection paused)
    ret_cam, frame_cam = cap_cam.read()
    if not ret_cam:
        break

    keypoints = []
    if detection_enabled:
        results = model(frame_cam, verbose=False)
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []
    else:
        keypoints = []

    if len(keypoints) > 0:
        person = keypoints[0]
        if prev_keypoints is not None and person.shape == prev_keypoints.shape:
            pixel_dist = np.linalg.norm(person - prev_keypoints, axis=1).mean()
            dt = time.time() - prev_time
            pixel_speed = pixel_dist / dt if dt > 0 else 0
            speed_buffer.append(pixel_speed)
            avg_speed = np.mean(speed_buffer)
            smoothed_speed = smoothing * smoothed_speed + (1 - smoothing) * avg_speed
            normalized_speed = np.clip(smoothed_speed / max_speed, 0, 1)
            speed_factor = 1.0 + 9.0 * normalized_speed
            delay = 0.04 / speed_factor
            paused = normalized_speed < movement_threshold

            if not paused:
                h_cam, w_cam, _ = frame_cam.shape
                dist_m = pixels_to_meters(pixel_dist, w_cam, camera_fov_deg, subject_distance_m)
                speed_m_s = dist_m / dt if dt > 0 else 0
                total_dist_m += dist_m

                for _ in range(int(speed_factor)):
                    ret_video, frame_video = cap_video.read()
                    if not ret_video:
                        cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                if frame_video is not None:
                    # always draw skeleton on pose subframe
                    pose_sub = frame_cam.copy()
                    draw_skeleton(pose_sub, person, skeleton_pairs, (0, 255, 0))

                    # optionally draw skeleton on main video
                    if draw_on_video:
                        h_cam, w_cam, _ = frame_cam.shape
                        h_vid, w_vid, _ = frame_video.shape
                        scale_x = w_vid / w_cam
                        scale_y = h_vid / h_cam
                        for (a, b) in skeleton_pairs:
                            if a < len(person) and b < len(person):
                                xa, ya = person[a]
                                xb, yb = person[b]
                                cv2.line(frame_video,
                                         (int(xa * scale_x), int(ya * scale_y)),
                                         (int(xb * scale_x), int(yb * scale_y)),
                                         (0, 255, 0), 2)

                    show_video_frame(frame_video.copy(), pose_subframe=pose_sub)
                time.sleep(max(0.0, float(delay)))
            else:
                if frame_video is not None:
                    pose_sub = frame_cam.copy()
                    draw_skeleton(pose_sub, person, skeleton_pairs, (0, 255, 0))
                    show_video_frame(frame_video.copy(), pose_subframe=pose_sub)

        prev_keypoints = person
        prev_time = time.time()

    else:
        # no keypoints: still show UI using last video frame_video
        if frame_video is not None:
            pose_sub = frame_cam.copy()
            if prev_keypoints is not None:
                draw_skeleton(pose_sub, prev_keypoints, skeleton_pairs, (0, 255, 0))
            show_video_frame(frame_video.copy(), pose_subframe=pose_sub)

    # === Key handling ===
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        if overlay_open:
            close_overlay()
        else:
            break
    elif key == ord('m'):
        menu_open = not menu_open
        menu_target = 1.0 if menu_open else 0.0
    elif key == ord('p'):
        show_pose = not show_pose
        target_alpha_pose = 1.0 if show_pose else 0.0
    elif key == ord('h'):
        if overlay_open and overlay_mode == "help":
            close_overlay()
        else:
            open_overlay("help")
    elif key == ord('i'):
        # Toggle motion info with smooth fade
        show_motion_info = not show_motion_info
        print(f"Motion info {'shown' if show_motion_info else 'hidden' }.")
    elif key == ord('a'):
        if overlay_open and overlay_mode == "about":
            close_overlay()
        else:
            open_overlay("about")

# cleanup
cap_cam.release()
cap_video.release()
cv2.destroyAllWindows()
