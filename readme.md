# 九数：

- 要加环境变量：`export HF_MODULES_CACHE=~/.cache/huggingface`，否则 `/work`目录没有权限。
- sh添加权限： `chmod u+x xxx.sh`

# 运行

```
nohup fine-tuning.sh experiments/outputs/log 2>1 &
```
