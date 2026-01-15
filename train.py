import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('cfg/yolo26n.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='datasets/rgbt3m_tinyfire_enhance.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=64,
                close_mosaic=10, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
                workers=2, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
                optimizer='SGD', # using SGD 新增MuSGD优化器
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,不懂就在百度云.txt找断点续训的视频
                # amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/rgbt3m_tinyfire_enhance',
                name='rgbt3m_tinyfire_enhance-yolo26n-200epochs-',
                )
