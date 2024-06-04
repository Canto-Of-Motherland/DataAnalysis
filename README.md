# 使用说明
1. 将model_script文件夹置于DataAnalysis/DataAnalysisApp文件夹下；
2. 将对应的模型参数（由于模型参数过大无法上传，请通过lzx2562521178@outlook.com请求）文件夹（data和model文件夹）放置于model_script文件夹下；
3. 运行即可。
# 特别提醒
1. 如果仅仅想看演示效果，请不要执行使用说明中的操作，而是将DataAnalysis/DataAnalysisApp文件夹下的views.py文件中第162~177行注释，并取消153~161行的注释；
2. 舆情分析部分的后端代码参见https://github.com/Erutaner/weibo_text_toolkit，为大创产出成果之一；
3. 舆情分析的后端由于涉及到预训练的LLM和双向多层LSTM，因此参数量极大，运行该模型需要一张以上的4090才能保证运算时间，因此代码中使用预置数据。
4. 如果您认为您具备运行上面若干个模型的能力（句分析需要RTX 3070及以上，舆情分析需要RTX 4090及以上），鄙人愿提供相关的帮助和文件支持
