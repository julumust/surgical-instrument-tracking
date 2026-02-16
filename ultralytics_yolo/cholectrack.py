from ultralytics import YOLO

# Load a model, in this case, we will use yolo26m.pt
model = YOLO("yolo26m.pt")

model = model.train(
    data="cholectrack.yaml",
    epochs=100, # Total number of training epochs, each epoch represents a full pass over the entire dataset
    imgsz=640, # Target image size for training
    device=0, # Specifies the computational device for training, here we use an RTX 3070Ti

    # Augmentation and Hyperparameters
    hsv_h= # Adjusts image hue, colour variability helps the model generalise across different lighting conditions 
)

