# Setup Instructions

## 1. Create a Virtual Environment
```bash
python -m venv venv
```

## 2. Activate the Virtual Environment
### Windows
```bash
venv\Scripts\activate
```
### Linux/macOS
```bash
source venv/bin/activate
```

## 3. Clone the Repository
```bash
git clone https://github.com/Joshuakevin23/cv
cd cv
```

## 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 5. Download the Model
Download the yolo-crowd.pt model file using the provided link and place it in the project root directory.
The link is [yolo-crowd.pt](https://drive.google.com/file/d/1xxXVCzseuzmHv7NoMQ03RVU_tDisWXjM/view?usp=sharing)

## 6. Run the Application
```bash
streamlit run app.py
```

## Benchmarking
The default video path is configured in:
`scratch/benchmark_memory.py`

To use a different video, update the video path in that file.

Run the benchmark with a frame skip count:
```bash
python scratch/benchmark_memory.py 5
```
Where 5 is the frame skip count.
