# AutoPPT
 <img src="./figure_for_readme/autoppt_tieser.PNG" alt="Italian Trulli" 
 width="800" 
 height="400">

<br>It's again the day before reporting to your boss. Hundreds of thousands of tables and figures are smiling at you, evilly. You pray for rescue; just at that moment, suddenly, you reach this page. Congratulations! You've found AutoPPT, which ganna be your sharpest Knife for experiment reports of all time; following the synopsis below, it'll lead you to a new world. AutoPPT will save your hand from the destruction caused by copy and paste.</br>

# synopsis
 <img src="./figure_for_readme/LOGO.PNG" alt="Italian Trulli" 
 width="200" 
 height="100">
 
<br>-Requirement：</br>
`conda install -c conda-forge python-pptx` 

-Usage：

1.Create the template silde. (Please take the example template "template.pptx" as reference.)

‧如何設置變數(placeholder)：輸入 #(你的變數名稱) 於對應區塊。
‧新增圖片變數：使用圖形工具拉一個矩形，矩形中面設置變數名稱。
‧新增文字變數：使用文字工具拉一個文字方塊，方塊中設置變數名稱。

2. Import ppt_recorder from auto_ppt_module.py and load the template silde. example:
<code>
from auto_ppt_module import ppt_recorder
import numpy as np
import io
from PIL import Image
import copy
# Create ppt recorder given the reference template.
writer = ppt_recorder(template='template_example.pptx')
# Get the placeholders that you've created in the template file. 
# Placeholders are formated in dictionary, in which the keys are repected to their name.
ph = writer.placeholder()
test_image1 = np.eye(128)
test_image2 = np.random.uniform(low=0.0, high=1.0, size=[64,64,3])
for i in range(3):
    # Duplicate a set of formate respected to your template pptx file.
    # You must include .new_record() in your code even if only one set of your template is needed in your report.
    writer.new_record()
    # Create a placeholder-data dictionary.
    feed_dict={ph['text1']:'Hello World!_'+str(i), 
             ph['pic1']:test_image1,
             ph['pic2']:test_image2,
             ph['text2']:"What's up, World?_"+str(i),
             ph['pic3']:test_image1,
             ph['pic4']:test_image2}

    # Assign your data to pptx file.
    writer.assign(feed_dict=feed_dict)
#export pptx file.
writer.to_pptx('result.pptx')
</code>
‧文字輸入格式：str
‧圖片輸入格式：3D numpy array (W,H,C), which scaled in [0,1].
