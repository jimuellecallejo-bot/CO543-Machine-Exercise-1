import cv2
import numpy as np
import time
import random
from ultralytics import YOLO

# Baseline Parameters

green_move_threshold = 0.04
red_move_threshold_base = 0.055
red_grace_ms = 650
idle_warning_ms = 1800
idle_death_ms = 3600
green_duration_range = (2600, 4200)
red_duration_range = (1700, 2900)

# State Machine

STATE_GREEN = "GREEN"
STATE_RED = "RED"
STATE_WARNING = "WARNING"
STATE_DEAD = "DEAD"

# YOLOv8 Model

model = YOLO("yolov8n.pt")  # lightweight model

# Initalizaiton

cap = cv2.VideoCapture(0)
prev_gray = None
state = STATE_GREEN
idle_start = None
last_switch_time = time.time()
green_duration = random.randint(*green_duration_range) / 1000.0
red_duration = random.randint(*red_duration_range) / 1000.0
level = 1
cycle = 1

# Motion Detection

def compute_motion_score(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    motion_score = np.mean(diff) / 255.0
    return motion_score

# State Indicators

def switch_state(new_state):
    global state, last_switch_time, idle_start
    state = new_state
    last_switch_time = time.time()
    if new_state == STATE_WARNING:
        idle_start = last_switch_time
    else: 
        idle_start = None
    print(f"STATE -> {state}")

def next_cycle():
    global cycle, level
    cycle += 1
    if cycle > 3:  # 3 cycles per level 
        level += 1
        cycle = 1
        print(f"LEVEL UP -> {level}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_gray is None:
        prev_gray = gray
        continue

    motion_score = compute_motion_score(prev_gray, gray)
    prev_gray = gray

    now = time.time()
    elapsed = (now - last_switch_time)

    # YOLO person detection
    results = model(frame, verbose=False)
    verified_person = False
    for box in results[0].boxes:
        if int(box.cls[0]) == 0: # class 0 = person

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            verified_person = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            break

    # GREEN State

    if state == STATE_GREEN:
        if motion_score < green_move_threshold:
            if idle_start is None:
                idle_start = now
            idle_elapsed = (now - idle_start) * 1000
            if idle_elapsed > idle_warning_ms and state != STATE_WARNING:
                switch_state(STATE_WARNING)
            if idle_elapsed > idle_death_ms:
                switch_state(STATE_DEAD)
        else:
            idle_start = None # reset idle timer

        if elapsed > green_duration:
            switch_state(STATE_RED)
            red_duration = random.randint(*red_duration_range) / 1000.0

    # WARNING State
    
    elif state == STATE_WARNING: # Ensure idle_start is initialized
        if idle_start is None:
            idle_start = now
        
        idle_elapsed = (now - idle_start) * 1000

        if motion_score >= green_move_threshold:
            idle_start = None
            switch_state(STATE_GREEN)
        elif idle_elapsed > idle_death_ms:
                switch_state(STATE_DEAD)

        cv2.putText(frame, "Keep moving!", (100, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            
    # RED State
    
    elif state == STATE_RED:
        if elapsed * 1000 > red_grace_ms:
            if motion_score > red_move_threshold_base * (1 + 0.01 * level):
                switch_state(STATE_DEAD)

        if elapsed > red_duration:
            next_cycle()
            switch_state(STATE_GREEN)
            green_duration = random.randint(*green_duration_range) / 1000.0

    # DEAD State
    
    elif state == STATE_DEAD:
        cv2.putText(frame, "YOU ARE DEAD", (100, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(frame, "PRESS 'r' TO RESTART GAME", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(frame, "PRESS 'q' TO QUIT GAME", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.imshow("Red Light Green Light (RLGL)", frame)
        key = cv2.waitKey(0)
        if key == ord('r'):  # Restart
            level, cycle = 1, 1
            switch_state(STATE_GREEN)
            green_duration = random.randint(*green_duration_range) / 1000.0
        elif key == ord('q'):
            break
        continue

# Phase Time Remaining

    if state == STATE_GREEN or state == STATE_WARNING:
        phase_duration = green_duration
    elif state == STATE_RED:
        phase_duration = red_duration
    else:
        phase_duration = 0
    
    remaining_time = max(0, phase_duration - elapsed)
    remaining_text = f"TIME LEFT: {remaining_time: .2f}s"

    # UI Overlay

    state_text_colors = {
        STATE_GREEN: (0, 255, 0),
        STATE_RED: (0, 0, 255),
        STATE_WARNING: (0, 255, 255),
        STATE_DEAD: (255, 255, 255)
    }

    cv2.putText(frame, f"State: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, state_text_colors[state], 3)
    cv2.putText(frame, f"Motion Score: {motion_score:.3f}", (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Verified Person: {verified_person}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if phase_duration > 0:
        frame_height, frame_width = frame.shape[:2]
        (text_width,text_height), _ = cv2.getTextSize(remaining_text,
                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.9, 2)
        x = frame_width - text_width - 10
        y = 30 #top margin

        cv2.putText(frame, remaining_text, (400, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        level_text = f"LEVEL: {level} "
        cv2.putText(frame, level_text, (480, y + text_height + 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cycle_text = f"CYCLE: {cycle}/3 "
        cv2.putText(frame, cycle_text, (480, y + text_height + 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    cv2.imshow("Red Light Green Light (RLGL)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()