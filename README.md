# rocket-recycling

本项目基于 https://github.com/jiupinjia/rocket-recycling 中提供的火箭环境。

此项目实现了8种RL算法，以供大家学习，并基于原环境提出了5个改进的点。

其中包括：
- 添加风速 `rocket1.py`
- 添加燃料因素 `rocket2.py`
- 模拟月球环境 `rocket3.py`
- 扩展多个状态输入，并构建了一个简单的state attention module
- 给softmax添加随episode增大而减小的温度。使得前期注重探索，后期注重利用。

# 代码运行方法
运行每个算法：直接找的对应名称的py文件，运行即可。

运行改进算法： `a2c.py`  `a2c_state.py` `a2c_attention.py`  `a2c_temperature.py  `

运行其他环境：
`from rocket import Rocket `
改成：
`from rocket1 import Rocket` 
`from rocket2 import Rocket` 
`from rocket3 import Rocket` 
`from rocket4 import Rocket` 
即可


更改不同任务：
```python
task = 'landing'  # 'hover' or 'landing'
```

生成文件位置：
```python
env = Rocket(task=task, max_steps=max_steps)
ckpt_folder = os.path.join('./', task + '_rocket' + '_ckpt')
if not os.path.exists(ckpt_folder):
    os.mkdir(ckpt_folder)
```

视频文件的生成与保存：
```python
if episode_id % 100 == 1 and render==True:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(os.path.join(ckpt_folder, str(episode_id).zfill(8) + '_' +task +'.mp4'), fourcc, 200.0,(768,768))
    for f in frame_list:
        out.write(f)
    out.release()
```

测试代码：
`example_inference.py`


