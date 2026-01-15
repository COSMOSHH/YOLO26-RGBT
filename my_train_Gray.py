import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('cfg/yolo11n-gray.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'datasets/rgbt3m_tinyfire_enhance_single_ir.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=32,
                close_mosaic=10,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="Gray",  # Gray16bit
                channels=1,
                project='runs/rgbt3m_tinyfire_enhance_single_ir',
                name='rgbt3m_tinyfire_enhance_single_ir-yolo11n-gray-',
                )