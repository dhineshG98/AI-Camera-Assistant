# AI Camera Assistant

AI Camera Assistant is a computer vision project designed to help visually impaired users identify objects around them using voice feedback.

## Features

- Real-time object detection using YOLOv8
- Voice commands using Vosk Speech Recognition
- Audio feedback using Edge TTS
- Object position detection (Left, Center, Right)
- Simple GUI using Tkinter
- Webcam-based environment awareness

## Technologies Used

- Python
- YOLOv8
- OpenCV
- Vosk
- Edge TTS
- Pygame
- Tkinter

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Camera-Assistant.git
cd AI-Camera-Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Vosk Model

Download the English Vosk model from:

https://alphacephei.com/vosk/models

Extract the model folder into the project directory.

### 4. Run the project

```bash
python main.py
```

## Voice Commands

| Command | Action |
|----------|----------|
| start | Start object detection |
| stop | Pause object detection |
| stop detection | Pause object detection |
| stop camera | Close the application |

## Project Structure

```text
AI-Camera-Assistant/
│
├── main.py
├── requirements.txt
└── README.md
```

## How It Works

1. Captures video from the webcam.
2. Detects objects using YOLOv8.
3. Determines object location (left, center, right).
4. Converts detected information into speech.
5. Provides audio assistance to the user.

## Future Enhancements

- Distance estimation
- Currency recognition
- Face recognition
- Navigation assistance
- Mobile application support

## Author

Dhinesh G

## License

This project is created for educational and research purposes.
