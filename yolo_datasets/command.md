# 训练
python train.py --weights /home/ai/gsh/Code/yolov5-6.2/models/yolov5s.pt --cfg /home/ai/gsh/Code/

yolov5-6.2/yolov5-6.2/yolov5-6.2/models/yolov5s.yaml --data /home/ai/gsh/Code/yolov5-6.2/yolov5-6.2/

yolov5-6.2/data/firebox_withonline_trainval.yaml --epochs 100

# 测试
`python detect.py --weights /home/ai/gsh/Code/yolov5-6.2/yolov5-6.2/yolov5-6.2/runs/train/exp15/`

`weights/best.pt --source /home/ai/gsh/online_test/1.jpg`

`watch -n 0.5 nvidia-smi`

# done
`exp_5m_firebox_withonline_withweights_epochs100`

`exp_5s_firebox_withonline_withweights_epochs100`

`exp_5l_firebox_withonline_withweights_epochs100`

`exp_5n_firebox_withonline_withweights_epochs100`

`exp_5m_firebox_epochs100`

`exp_5m_firebox_withonline_epochs100`

`exp_5m_firebox_withweights_epochs100`

`exp_5l_firebox_withweights_epochs100`

`exp_5s_firebox_withweights_epochs100`

`exp_5n_firebox_withweights_epochs100`

`exp_5n_firebox_withonline_epochs300`


# todo
