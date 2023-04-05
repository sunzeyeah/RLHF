
为验证不同预训练模型使用deepspeed的训练效率是否能达到官方宣称的效果（加速、节省GPU等），进行了benchmarking
- 实验场景：SFT阶段训练
- 实验数据：SFT & Reward Data的验证集，共1万条样本
- 实验参数：```batch_size=1, max_sequence_length=512, gradient_checkpointing=False```


<table>
    <thead>
        <tr> <td>模型</td>  <td>整体耗时/epoch</td>  <td>单条样本耗时</td>  <td>内存使用量</td>  <td>显存使用量</td>  <td>GPU型号和数量</td> <td>fp16</td> <td>bf16</td> <td>deepspeed stage</td> <td>offload optimizer</td> <td>pin memory</td> <td>offloard param</td> <td>overlap comm</td> <td>allgather bucket size</td> <td>stage3 max live parameters</td> </tr>
    </thead>
   <tbody>
       <tr> <td rowspan="15">Pangu-350M</td>  <td>20min</td>  <td>1.17s/it</td>  <td></td>  <td>1*8750MB</td>  <td>1*V100 16G</td>  <td>false</td>  <td>-</td>  <td>-</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td>19min</td>  <td>1.03s/it</td>  <td></td>  <td>1*9010MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>-</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td>7.5min</td>  <td>1.10s/it</td>  <td></td>  <td>3*9406MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>0</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td>6.5min</td>  <td>1.05s/it</td>  <td></td>  <td>3*5674MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>1</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td>14min</td>  <td>2.10s/it</td>  <td></td>  <td>3*6262MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>false</td> <td>-</td> <td>-</td> <td>false</td> <td>5e8</td> <td>-</td> </tr>
       <tr> <td>18min</td>  <td>2.6s/it</td>  <td>18G</td>  <td>3*3668MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>2e8</td> <td>-</td> </tr>
       <tr> <td>18.5min</td>  <td>2.65s/it</td>  <td>18G</td>  <td>3*4240MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>5e8</td> <td>-</td> </tr>
       <tr> <td>18.5min</td>  <td>2.65s/it</td>  <td>18G</td>  <td>3*5194MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>1e9</td> <td>-</td> </tr>
       <tr> <td>19min</td>  <td>2.80s/it</td>  <td>18G</td>  <td>3*12824MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>5e9</td> <td>-</td> </tr>
       <tr> <td>47min</td>  <td>6.75s/it</td>  <td>14G</td>  <td>3*4914MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>false</td> <td>-</td> <td>false</td> <td>false</td> <td>1e9</td> <td>-</td> </tr>
       <tr> <td>3.3h</td>  <td>29s/it</td>  <td>18G</td>  <td>3*3512MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>false</td> <td>false</td> <td>1e9</td> <td>-</td> </tr>
       <tr> <td>4h</td>  <td>34s/it</td>  <td>24G</td>  <td>3*3466MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e9</td> <td>-</td> </tr>
       <tr> <td>3.8h</td>  <td>33s/it</td>  <td>24G</td>  <td>3*3746MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>true</td> <td>1e9</td> <td>-</td> </tr>
       <tr> <td>3.8h</td>  <td>33s/it</td>  <td>24G</td>  <td>3*3594MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>true</td> <td>5e8</td> <td>-</td> </tr>
       <tr> <td>4h</td>  <td>35s/it</td>  <td>24G</td>  <td>3*3526MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>true</td> <td>2e8</td> <td>-</td> </tr>
       <tr> <td rowspan="10">Pangu-2.6B</td>  <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>false</td>  <td>-</td>  <td>-</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>-</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>0</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>1</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>false</td> <td>-</td> <td>-</td> <td>false</td> <td>5e8</td> <td>-</td> </tr>
       <tr> <td>1.5h</td>  <td>12.3s/it</td>  <td>59G</td>  <td>3*10796MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>2e8</td> <td>-</td> </tr>
       <tr> <td>1.3h</td>  <td>12s/it</td>  <td>59G</td>  <td>3*11368MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>5e8</td> <td>-</td> </tr>
       <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>false</td> <td>-</td> <td>false</td> <td>false</td> <td>-</td> <td>1e9</td> </tr>
       <tr> <td>7.5h</td>  <td>64.5s/it</td>  <td>58G</td>  <td>3*13428MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>false</td> <td>false</td> <td>-</td> <td>1e9</td> </tr>
       <tr> <td>11.3h</td>  <td>95s/it</td>  <td>109G</td>  <td>3*12170MB</td>  <td>3*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>-</td> <td>1e9</td> </tr>
       <tr> <td rowspan="17">Pangu-2.6B</td>  <td></td>  <td>1.32s/it</td>  <td></td>  <td>1*49347MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>false</td>  <td>-</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td></td>  <td>1.27s/it</td>  <td></td>  <td>1*52783MB</td>  <td>1*A100 80G</td>  <td>true</td>  <td>false</td>  <td>-</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td></td>  <td>1.27s/it</td>  <td></td>  <td>1*52783MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>-</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td></td>  <td>-</td>  <td></td>  <td>CUDA Error</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>0</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td></td>  <td>-</td>  <td></td>  <td>CUDA Error</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>1</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> </tr>
       <tr> <td></td>  <td>-</td>  <td></td>  <td>CUDA Error</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>2</td>  <td>false</td> <td>-</td> <td>-</td> <td>false</td> <td>5e8</td> <td>-</td> </tr>
       <tr> <td></td>  <td>9-11s/it</td>  <td></td>  <td>1*12537MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>2e8</td> <td>-</td> </tr>
       <tr> <td></td>  <td>8-9s/it</td>  <td></td>  <td>1*13539MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>5e8</td> <td>-</td> </tr>
       <tr> <td></td>  <td>9-11s/it</td>  <td></td>  <td>1*15041MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>1e9</td> <td>-</td> </tr>
       <tr> <td></td>  <td>8-9s/it</td>  <td></td>  <td>1*14887MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>true</td> <td>5e8</td> <td>-</td> </tr>
       <tr> <td></td>  <td>-</td>  <td></td>  <td>CUDA Error</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>3</td>  <td>false</td> <td>-</td> <td>false</td> <td>false</td> <td>5e8</td> <td>1e9</td> </tr>
       <tr> <td></td>  <td>17-18s/it</td>  <td></td>  <td>1*16935MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>3</td>  <td>true</td> <td>true</td> <td>false</td> <td>false</td> <td>-</td> <td>1e9</td> </tr>
       <tr> <td></td>  <td>20-21s/it</td>  <td></td>  <td>1*12219MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>-</td> <td>1e9</td> </tr>
       <tr> <td></td>  <td>19-20s/it</td>  <td></td>  <td>1*15981MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>true</td> <td>-</td> <td>1e9</td> </tr>
       <tr> <td></td>  <td>22-23s/it</td>  <td></td>  <td>1*12023MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>true</td> <td>-</td> <td>5e8</td> </tr>
       <tr> <td></td>  <td>20-21s/it</td>  <td></td>  <td>1*12023MB</td>  <td>1*A100 80G</td>  <td>false</td>  <td>true</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>true</td> <td>-</td> <td>2e8</td> </tr>
    </tbody>
</table>
