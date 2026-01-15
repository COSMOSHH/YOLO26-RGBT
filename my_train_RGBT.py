import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("cfg/yolo26n-RGBT-midfusion.yaml")
    model.train(
        data=R"datasets/rgbt3m_tinyfire_enhance.yaml",
        cache=False,
        imgsz=640,
        epochs=200,
        batch=32,
        close_mosaic=10,
        workers=2,
        device="0",
        optimizer="SGD",  # using SGD
        # resume=False,
        # lr0=0.002,
        # resume='', # last.pt path
        # amp=False, # close amp
        # fraction=0.2,
        # pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
        use_simotm="RGBT",
        channels=4,
        project="runs/rgbt3m_tinyfire_enhance",
        name="rgbt3m_tinyfire_enhance-yolo26n-RGBT-midfusion-200epochs-",
    )
