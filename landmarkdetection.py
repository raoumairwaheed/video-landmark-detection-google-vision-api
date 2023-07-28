from google.cloud import videointelligence_v1 as videointelligence
from google.cloud import vision
import os
import io
import cv2
import json
import logging

# Replace with your file, downloaded from Google Cloud Video Intelligence API
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'loyal-vent-356807-b0c9c97dce30.json'


def analyze_shots(path, video):
    fps = float(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS: ",fps)
    print("Total Frames: ", total_frames)

    # Read the video file as binary and convert to bytes
    with io.open(path, "rb") as f:
        input_content = f.read()

    # Use the VideoIntelligenceServiceClient from videointelligence_v1
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]

    # Use annotate_video method from videointelligence_v1, not VideoIntelligenceServiceClient
    operation = video_client.annotate_video(
        request={"features": features, "input_content": input_content}
    )

    print("\nProcessing video for shot change annotations:")

    # result method is not available in videointelligence_v1
    result = operation.result(timeout=90)

    print("\nFinished processing.")

    shot_frames = []  # Store the frame numbers where shot changes occur

    # Iterate through the shot annotations and store the frame numbers
    for i, shot in enumerate(result.annotation_results[0].shot_annotations):
        start_time = (
                shot.start_time_offset.seconds + shot.start_time_offset.microseconds / 1e6
        )
        start_frame = int(start_time * fps)

        # Store the first 10 frames of each shot change
        for j in range(10):
            frame_number = start_frame + j
            if frame_number < total_frames:
                shot_frames.append(frame_number)

    print("Saved Frames after Shot Change Detection: ", shot_frames)
    return shot_frames


def get_approximate_bounding_box(vertices):
    # Calculate an approximate bounding box based on the vertices of the bounding polygon
    x_coordinates = [vertex.x for vertex in vertices]
    y_coordinates = [vertex.y for vertex in vertices]
    min_x, min_y = min(x_coordinates), min(y_coordinates)
    max_x, max_y = max(x_coordinates), max(y_coordinates)

    box_width = max_x - min_x
    box_height = max_y - min_y

    # Adjust the values as needed to fit the bounding box size you desire
    padding = 10
    vertices = [
        (min_x - padding, min_y - padding),
        (max_x + padding, min_y - padding),
        (max_x + padding, max_y + padding),
        (min_x - padding, max_y + padding)
    ]
    return vertices


def detect_landmarks(video, shot_frames):
    client = vision.ImageAnnotatorClient()
    # Create an empty dictionary to store landmark information for each frame
    frame_landmarks = {}
    try:
        for current_frame in shot_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = video.read()
            if not ret:
                continue

            cv2.putText(frame, "Shot Change", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, encoded_frame = cv2.imencode('.jpg', frame)
            content = encoded_frame.tobytes()
            image = vision.Image(content=content)

            response = client.landmark_detection(image=image)
            landmarks = response.landmark_annotations
            # Create an empty list to store detected landmarks for the current frame
            detected_landmarks = []
            for landmark in landmarks:
                for location in landmark.locations:
                    lat_lng = location.lat_lng

                    print(f"Frame: {current_frame}")
                    print(f"Landmark: {landmark.description}")
                    print(f"Latitude: {lat_lng.latitude}")
                    print(f"Longitude: {lat_lng.longitude}")

                    # Get the bounding box vertices
                    if landmark.bounding_poly.vertices:
                        vertices = get_approximate_bounding_box(landmark.bounding_poly.vertices)
                    else:
                        # If there are no vertices for the bounding box, set it to None
                        vertices = None
                    # Add detected landmark information to the list for the current frame
                    detected_landmarks.append({
                        'landmark_name': landmark.description,
                        'latitude': lat_lng.latitude if lat_lng else None,
                        'longitude': lat_lng.longitude if lat_lng else None,
                        'bounding_box': vertices
                    })

            if detected_landmarks:
                # Store the list of detected landmarks in the dictionary with the current frame number as the key
                frame_landmarks[current_frame] = detected_landmarks

    except KeyboardInterrupt:
        pass
    finally:
        video.release()
        cv2.destroyAllWindows()

        # Save the landmark information to a JSON file
        json_file = 'Landmarks.json'
        with open(json_file, 'w') as f:
            json.dump(frame_landmarks, f, indent=2)



if __name__ == "__main__":
    path = "sample_video.mp4"
    video = cv2.VideoCapture(path)

    try:
        shot_frames = analyze_shots(path, video)
        detect_landmarks(video, shot_frames)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        video.release()
