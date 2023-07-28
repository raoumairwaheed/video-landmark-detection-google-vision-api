# Video Landmark Detection Tool

This tool utilizes the Google Cloud Video Intelligence API and Google Cloud Vision API to detect landmarks in a video file. Due to the limitations of the Vision API, which operates on images rather than videos, it can be computationally costly to apply landmark detection on all frames of a video. To overcome this, the tool first performs shot change detection using the Video Intelligence API to identify different shots in the video. Then, for each shot, it applies landmark detection to the frames within that shot, reducing the processing load and making it more efficient.

> **Note**: Before using this tool, make sure to replace the `GOOGLE_APPLICATION_CREDENTIALS` environment variable with the path to your own Google Cloud Service Account key file (`loyal-vent-356807-b0c9c97dce30.json`) generated from the Google Cloud Console.

## Prerequisites

- Python 3.x installed
- OpenCV (cv2) library installed (`pip install opencv-python`)
- Google Cloud Video Intelligence API enabled
- Google Cloud Vision API enabled
- Google Cloud Service Account key file (JSON) with appropriate permissions

## Installation

1. Clone this repository to your local machine.
2. Install the required Python dependencies using the following command:

```bash
pip install opencv-python google-cloud-videointelligence google-cloud-vision
```

## Usage

To detect landmarks in a video file, use the following command in your terminal or command prompt:

```bash
python landmark_detection.py
```

The tool will process the video (`sample_video.mp4`) and generate a JSON file (`Landmarks.json`) containing the detected landmarks and their bounding boxes for each frame within the identified shots.

```python
# Replace these values with your desired video file and output JSON file paths
path = "sample_video.mp4"
video = cv2.VideoCapture(path)

try:
    shot_frames = analyze_shots(path, video)
    detect_landmarks(video, shot_frames)
except Exception as e:
    logging.error(f"Error occurred: {e}")
    video.release()
```

Please ensure that you have properly set up the Google Cloud Service Account key file and enabled the necessary APIs in your Google Cloud Console before running the tool.

## Output

After running the tool, the `Landmarks.json` file will contain the detected landmarks and their bounding boxes for each frame within the identified shots in the following format:

```json
{
    "frame_number": [
        {
            "landmark_name": "Eiffel Tower",
            "latitude": 48.8582,
            "longitude": 2.2945,
            "bounding_box": [
                [x1, y1],
                [x2, y2],
                [x3, y3],
                [x4, y4]
            ]
        },
        {
            "landmark_name": "Statue of Liberty",
            "latitude": 40.6892,
            "longitude": -74.0445,
            "bounding_box": [
                [x1, y1],
                [x2, y2],
                [x3, y3],
                [x4, y4]
            ]
        },
        ...
    ],
    "frame_number": [...]
}
```

The bounding box vertices represent the approximate bounding box around the detected landmark, and you can adjust the bounding box size by modifying the padding value in the `get_approximate_bounding_box` function. By utilizing shot change detection and applying landmark detection only on frames within each shot, the tool optimizes the processing of landmark detection in videos.