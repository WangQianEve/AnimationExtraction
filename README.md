python版本为2.7

1. 安装依赖包：运行
  sudo pip install -r requirements.txt

2. 缩放动画曲线提取

  输入视频要求前40帧为背景
  输入视频通过-v [path+filename]设定，默认是blue.avi
  输出路径通过-o [path]来设定，默认是./output/
  如果程序提示需要增大threshold，则通过-t [int]来设定，默认值是5，增加时一次加1，一般不建议加到10以上
  程序有两种策略，第一种是选择最大的一个区域，第二种是选择面积超过一定值的所有区域。默认是第二种策略，如果需要修改通过-l 1/0来设定。
  举例，运行指令
  python anim.py -v ./videos/white.avi -o ./output/ -t 7 -l 1

