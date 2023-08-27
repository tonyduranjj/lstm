# 使用 LSTM 进行多变量时间序列预测

在本教程中，我们将使用深度学习方法（LSTM）执行多变量时间序列预测。

我们先来了解两个话题

- 什么是时间序列分析？
- 什么是 LSTM？

## 时间序列分析：

时间序列表示基于时间顺序的一系列数据。它可以是秒、分钟、小时、天、周、月、年。未来的数据将取决于它以前的值

在现实世界中，我们主要有两种类型的时间序列分析

1. 单变量时间序列
2. 多变量时间序列

对于单变量时间序列数据，我们将使用单列进行预测

![image-20230408151000682](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230408151000682.png)

单变量时间序列数据

正如我们所看到的，只有一列，所以即将到来的未来值将仅取决于它以前的值

但是在多变量时间序列数据的情况下，我们将具有不同类型的特征值，并且目标数据将依赖于这些特征

![image-20230408151105826](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230408151105826.png)



多变量时间序列数据

正如我们在图片中看到的，在多变量中，我们将有多个这样的列来预测目标值（上图 “count” 是目标值）

在上面的图片数据中，计数值不仅取决于它以前的值，还取决于其他特征

**因此，要预测即将到来的计数值，我们必须考虑包括目标列在内的所有列，以预测目标值**

在执行多变量时间序列分析时，我们必须记住一件事。您可以看到，我们正在使用这些多个特征来对未来的目标数据进行预测，==因此当我们进行预测时，我们需要预测未来几天的特征值==（不包括目标列，因为我们将仅对此进行预测）

让我们用一个例子来理解

如果我们使用 5 列 [feature1， feature2， feature3， feature4， target] 来训练模型，那么我们需要为即将到来的预测日提供 4 列 [feature1， feature2， feature3， feature4]。

（如果您没有正确理解以上两段，请不要担心，在编码部分一切都会被清除）

## 长短期记忆

我不打算详细讨论 LSTM。我希望你们都知道它是什么。所以，我只给你一个简短的描述，如果你对LSTM没有太多了解，不用担心，我附上了一些重要的链接来研究 LSTM

LSTM 基本上是一个循环神经网络，能够处理长期依赖关系

假设您正在看电影。所以当那部电影里发生任何一种情况时，你已经知道之前发生了什么，也明白了，因为过去的原因，这种新情况发生了

RNN 也以相同的方式工作，它们记住过去的信息并使用它来处理当前的输入

RNN 的问题在于，由于梯度消失，它们无法记住长期依赖关系

因此，为了结局梯度消息和梯度爆炸，设计了 LSTM

如果您想了解更多关于 LSTM 的信息，请查看此链接 

1. https://youtu.be/rdkIOM78ZPk
2. https://medium.com/analytics-vidhya/in-depth-tutorial-of-recurrent-neural-network-rnn-and-long-short-term-memory-lstm-networks-3a782712a09f

让我们开始编码

让我们首先导入所需的库来进行预测

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
```

现在让我们使用 pandas 加载数据并检查输出

```python
df=pd.read_csv("train.csv",parse_dates=["Date"],index_col=[0])
df.head()
```

![image-20230408161435016](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230408161435016.png)

```python
df.tail()
```

![image-20230408161447924](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230408161447924.png)

好的，现在让我们花几秒钟来处理数据，正如我们所看到的，csv 文件包含 google 从 2001-01-25 到 2021-09-29 的股票数据

**这里我们试图预测“开盘价”列的未来值，所以“开盘价”是这里的目标列**

让我们检查一下数据的形状

```python
df.shape
```

数据的形状为 （5203，5）

现在让我们进行训练测试拆分，请记住，这里我们不能打乱数据，因为在时间序列中，我们必须遵循顺序

```python
test_split=round(len(df)*0.20)
df_for_training=df[:-1041]
df_for_testing=df[-1041:]
print(df_for_training.shape)
print(df_for_testing.shape)
```

df_for_training（4162， 5）df_for_testing（1041， 5）

如果您看到数据，我们可以注意到数据范围非常高，并且它们没有在同一范围内缩放，因此为了避免预测错误，让我们首先使用 MinMaxScaling 器缩放数据。

（您也可以使用标准缩放器）

```python
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)
df_for_training_scaled
```

![image-20230510144230298](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230510144230298.png)

万岁！我们的数据现在已归一化。

现在，让我们将数据拆分为 X 和 Y。

**这是最重要的部分，请正确阅读每个步骤**

```python
def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past : i, 0:dataset.shape[1]])
            dataY.append(dataset[i, 0])
    return np.array(dataX),np.array(dataY)

trainX, trainY = createXY(df_for_training_scaled, 30)
testX, testY = createXY(df_for_testing_scaled, 30)
```

让我们讨论一下我们在上面的代码中做了什么

我将讨论上述代码的 trainX 和 trainY，testX  和 testY 遵循相同的思路

==n_past 指的是我们要用多少历史步骤数来预测下一个目标值==

所以这里我使用了 30，表示我们将使用过去的 30 个值（具有包括目标列在内的所有特征）来预测第 31 个目标值

因此，在 trainX 中，我们将拥有所有特征值，而在 trainY 中，我们将只有目标值。

让我们详细说明该函数的每个部分 

dataset = df_for_training_scaled ， n_past = 30

当 i = 30 时：

data_X.append（df_for_training_scaled[ i - n_past ：i， 0 ：df_for_training.shape[1]]）

如您所见，从 n_past 开始的范围意味着 30，因此数据范围首次将是 [30 – 30 ： 30， 0 ： 5] 表示 [0：30 ，0 ： 5]，其实就是切片索引，逗号前表示行，逗号后表示列

因此，在 dataX 列表中，df_for_training_scaled[0：30，0：5] 数组将首次被添加，即数据的 0 - 29 行和 0 - 4 列将被添加

现在，dataY.append（df_for_training_scaled[i，0]）

如您所知，现在 [i, 0] = [30, 0] 表示数据中的第 30 行的第 0 列将被添加，由数据可知，第0 列即 Open 列

因此，前 30 行包含 5 列存储在 dataX 中，只有 Open 列的第 31 行存储在 dataY 中。然后我们将 dataX 和 dataY 列表转换为数组，因为我们需要它们以数组格式在 LSTM 中进行训练。

像这样，数据将被保存在 trainX和 trainY 中，直到数据集的长度。

让我们检查一下形状

```
print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
```

输出→ trainX 形状 — （4132， 30， 5） trainY 形状 — （4132，）

```
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)
```

输出→ testX 形状 — （1011， 30， 5）testY 形状 — （1011，）

由此可见，trainX 是一个三维数组，

```
print("trainX[0]-- \n",trainX[0])
print("trainY[0]-- ",trainY[0])
```

![image-20230510144221176](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230510144221176.png)

现在，如果您检查 trainX[1] 值，您会注意到它与 trainX[0] 中的数据相同，除了第一列，因为我们将看到前 30 列预测第 31 列，在第一次预测后它会自动移动到第二列并取下一个 2 值来预测下一个目标值。

让我们用一个简单的格式来解释这一切

trainX  →trainY

[0 ： 30，0：5] → [30，0]

[1：31， 0：5] → [31，0]

[2：32，0：5] →[32，0]

像这样，每个数据都将保存在 trainX 和 trainY 中

现在让我们训练模型，我用 girdsearchCV 做了一些超参数调优来找到基础模型。

```python
def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,5)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)
```

如果需要，您可以进行更多超参数优化，还可以添加更多层。如果数据集非常大，我建议您增加 LSTM 模型中的周期和单位。

您可以在第一个 LSTM 图层中看到输入形状为 （30，5）。它来自trainX形状。（trainX.shape[1]，trainX.shape[2]） → （30，5）

现在让我们将模型拟合到 trainX 和 trainY 数据中。

```python
grid_search = grid_search.fit(trainX,trainY)
```

运行需要一些时间，因为我们正在进行超参数调优。

你可以看到损失会像这样减少

![image-20230408220900431](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230408220900431.png)



现在让我们检查模型的最佳参数。

```python
grid_search.best_params_
```

输出为 — {'batch_size'： 20， 'epochs'： 10， 'optimizer'： 'adam'}

现在将最佳模型保存在my_model变量中。

```python
my_model=grid_search.best_estimator_.model
```

我们创建了模型来对时间序列进行预测。



现在，让我们使用测试数据集来测试模型。

```python
prediction=my_model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)
```

![image-20230408220914697](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230408220914697.png)

检查测试Y和预测的长度，这是相同的。现在让我们将 testY 与预测进行比较。

但是，这里有一个转折点，我希望你们记得我们在开始时缩放了数据，所以首先我们必须做一些反向缩放过程。

现在，反向缩放有点棘手，让我们通过代码查看所有内容

```python
scaler.inverse_transform(prediction)
```

![image-20230510144112282](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230510144112282.png)

如果您看到错误，则可以理解问题。在缩放数据时，我们每行有 5 列，现在我们只有 1 列是目标列。

所以我们必须改变形状才能使用inverse_transform

```
prediction_copies_array = np.repeat(prediction,5, axis=-1)
```

![image-20230510144124288](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230510144124288.png)

现在，如果您看到输出，您会意识到 5 列值是相似的。它只是无名无故地复制了 4 次单个预测列.

所以现在我们有 5 列 [同一列 5 次] .

现在让我们检查一下形状-

```python
prediction_copies_array.shape
```

输出→ （1011，5）

现在我们可以轻松使用inverse_transform函数。

```python
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),5)))[:,0]
```

我们只需要逆变换后的第一列，所以我们在最后使用了→ [：，0]。

现在我们需要将此 pred 值与 testY 进行比较。但是我们的测试Y也是可扩展的。因此，让我们使用与上述代码相同的逆变换。

```python
original_copies_array = np.repeat(testY,5, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),5)))[:,0]
```

现在让我们检查预测值和原始值→

```python
print("Pred Values-- " ,pred)
print("\nOriginal Values-- " ,original)
```

![image-20230510144206478](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230510144206478.png)

现在让我们做一个图来检查我们的pred和原始数据。

```python
plt.plot(original, color = 'red', label = 'Real Stock Price')
plt.plot(pred, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

![image-20230510144135695](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230510144135695.png)

呜呜！我们成功了。

好的，现在让我们再采取一些步骤。

到目前为止，我们训练了我们的模型，用测试值检查了该模型。现在让我们预测一些未来的值。

现在让我们从主 df 数据集中获取我们在开始时加载的最后 30 个值[为什么是 30？因为这是我们想要的过去值的数量，以预测第 31 个值]

```python
df_30_days_past=df.iloc[-30:,:]
df_30_days_past.tail()
```

![image-20230408221101613](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230408221101613.png)

我们可以看到我们拥有所有列，包括目标列（“打开”）

现在让我们预测未来的 30 个值

正如我之前告诉你们的，在多变量时间序列预测中，如果我们想通过使用不同的特征来预测单列，在进行预测时，我们需要特征值（目标列除外）来进行即将到来的预测。

所以，这里我们需要即将到来的 30 个值“最高价”、“最低价”、“收盘价”、“调整收盘价”列来预测“开盘价”列。

```python
df_30_days_future=pd.read_csv("test.csv",parse_dates=["Date"],index_col=[0])
df_30_days_future
```

![image-20230510144151309](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230510144151309.png)

如我们所见，我们拥有除 Open 列之外的所有列

现在，在使用模型进行预测之前，我们必须执行一些步骤

1. 我们必须扩展过去和未来的数据。正如您在我们未来的数据中看到的那样，我们没有 Open 列 ，所以在缩放它之前，只需在未来的数据中添加一个所有“0”值的 Open 列。
2. 缩放后，将以后数据中的 Open 列值替换为 “nan”
3. 现在将 30 天前的值与 30 天的新值附加（其中最后 30 个“打开”值是 nan）

```python
df_30_days_future["Open"]=0
df_30_days_future=df_30_days_future[["Open","High","Low","Close","Adj Close"]]
old_scaled_array=scaler.transform(df_30_days_past)
new_scaled_array=scaler.transform(df_30_days_future)
new_scaled_df=pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:,0]=np.nan
full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)
```

现在检查 full_df 。形状将为 （60，5），末尾的第一列有 30 个 nan 值。

现在要进行预测，我们必须再次使用 for 循环，这是我们在 trainX 和 trainY 中拆分数据时所做的。但是这次我们只有X，没有Y值

```python
full_df_scaled_array=full_df.values
all_data=[]
time_step=30
for i in range(time_step,len(full_df_scaled_array)):
    data_x=[]
    data_x.append(
     full_df_scaled_array[i-time_step :i , 0:full_df_scaled_array.shape[1]])
    data_x=np.array(data_x)
    prediction=my_model.predict(data_x)
    all_data.append(prediction)
    full_df.iloc[i,0]=prediction
```

让我们讨论上面的代码。对于第一个预测，我们有前 30 个值。这意味着，当 for 循环首次运行时，它会检查前 30 个值并预测第 31 个 Open 数据。

当第二个 for 循环将尝试运行时，它将跳过第一行并尝试获取接下来的 2 个值意味着 [30：1]，在这里我们将开始出现错误，因为在开放列的最后一行我们有“nan”，所以我们每次都用预测替换“nan”。

现在，让我们对预测进行逆变换，以检查实际值，就像我们之前所做的那样→

```python
new_array=np.array(all_data)
new_array=new_array.reshape(-1,1)
prediction_copies_array = np.repeat(new_array,5, axis=-1)
y_pred_future_30_days = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(new_array),5)))[:,0]
print(y_pred_future_30_days)
```

![image-20230408221309260](https://typora-ac999.oss-cn-shanghai.aliyuncs.com/image-20230408221309260.png)

干得好伙计们!!!!!!

我们预测了30个未来值