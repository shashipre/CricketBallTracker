import os
from ultralytics import YOLO

def main():
    # Load the YOLO model (YOLOv8n)
    model = YOLO("yolov8n.pt")  # Replace with the correct model path if necessary

    # Move model to GPU (cuda)
    model.to('cuda')  # Ensure you're using GPU, change to 'cpu' if needed

    # Path to the data.yaml file
    data_yaml_path = r"C:\Users\shash\OneDrive\Documents\Desktop\Sem-3\DP\Cricket PItch Detector\Ball-Tracking2-2\data.yaml"

    # Get the directory of the data.yaml file
    save_dir = os.path.dirname(data_yaml_path)

    # Train the model
    model.train(
        data=data_yaml_path,  # Path to your data.yaml file
        epochs=50,  # Number of epochs
        imgsz=640,  # Image size
        batch=30,  # Batch size
        workers=4,  # Number of data loading workers
        project="runs/train2",  # Output folder for training logs
        name="exp",  # Name of the experiment (folder)
        exist_ok=True  # Overwrite existing experiment folder
    )

    # Save the model in the same folder as the data.yaml
    model.save(os.path.join(save_dir, 'best_model2.pt'))  # Save the model in the same directory as data.yaml


if __name__ == '__main__':
    main()
