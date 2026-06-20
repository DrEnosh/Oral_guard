import multiprocessing

def main():
    from ultralytics import YOLO, settings
    settings.update({"mlflow": False})
    
    model = YOLO(r"C:\Users\enosh\oralguard\src\detector\weights\oralguard_det\weights\best.pt")
    model.train(
        data=r"C:\Users\enosh\oralguard\data\combined\dental_combined.yaml",
        epochs=1,
        imgsz=1024,
        batch=8,
        device=0,
        workers=2, # Test if workers > 0 is supported on Windows
        project=r"C:\Users\enosh\oralguard\src\detector\weights",
        name="test_workers",
        exist_ok=True,
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
