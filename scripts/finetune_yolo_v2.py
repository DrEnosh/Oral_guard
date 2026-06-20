"""STEP 5: Fine-tune YOLO detector v2 on expanded combined dataset."""
import multiprocessing

def main():
    from ultralytics import YOLO, settings
    import mlflow

    # Disable built-in MLflow logging to prevent SQLite parenthesized metric crashes
    settings.update({"mlflow": False})

    print("Loading pre-trained OralGuard YOLO model...")
    model = YOLO(r"C:\Users\enosh\oralguard\src\detector\weights\oralguard_det\weights\best.pt")

    print("Starting fine-tuning on expanded combined dataset (5 epochs, 2 dataloader workers)...")
    results = model.train(
        data=r"C:\Users\enosh\oralguard\data\combined\dental_combined.yaml",
        epochs=5,
        imgsz=1024,
        batch=8,
        device=0,
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=3,
        patience=10,
        save=True,
        project=r"C:\Users\enosh\oralguard\src\detector\weights",
        name="oralguard_finetuned_v2",
        pretrained=True,
        exist_ok=True,
        verbose=True,
        workers=2,  # Faster data loading
    )

    print("\n=== YOLO V2 RESULTS ===")
    metrics = results.results_dict
    print("Overall mAP50:", metrics.get("metrics/mAP50(B)", "N/A"))
    print("Overall mAP50-95:", metrics.get("metrics/mAP50-95(B)", "N/A"))

    for key, val in sorted(metrics.items()):
        print(f"  {key}: {val}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
