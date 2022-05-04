# TyTalk
Convert Chinese Mandarin texts to vsqx files for Vocaloid, automatically predicting the prosody.
依据普通话文本合成vsqx文件，自动进行韵律（时长、语调）预测。

除分词外，完全基于人工规则，不包含机器学习成分。

依赖：jieba、pypinyin

本项目由文科生开发，欢迎拍砖。编程规范上面多有疏漏，但格式已经尽量符合PEP8标准。

分词器默认为 jieba ，也可在 reliance 中新建 baiduClient.py 脚本，以 client() 函数返回一个百度NLP的 client 对象。再把 main.py 中 (if name == main) 下的 Talker的参数改为 online=True ，即可使用百度NLP的分词和句法分析功能。
如您所见，本软件还预置了基于句法结构的停顿预测功能，但效果并不稳定。在完成上面的操作之后，这一功能也会打开。

本项目（TyTalk）是专用于vsqx文件生成的，其核心代码也可以用于通用的韵律预测。这一通用的预测功能属于其父项目AutoTalk（暂未发布）。
# Version 版本
此项目中的版本为0.6版本

# Latest Version 最近版本
TyTalk已被整合进入[VvTalk](https://github.com/GalaxieT/VvTalk)项目中。
