# Kaggle 房价预测

本仓库是 Kaggle House Prices 房价预测任务的 PyTorch 实现。项目使用房屋结构、面积、质量、位置等表格特征训练回归模型，预测测试集中每套房屋的最终成交价 `SalePrice`，并生成 Kaggle 可提交的 `submission.csv`。

> 说明：仓库名是 `kaggle-hous-price-forecast`，其中 `hous` 少了一个 `e`；README 按当前仓库名和项目内容说明。

## 文件说明

- `kaggle_house_price_pred.py`：主脚本，包含数据读取、特征预处理、模型定义、K 折交叉验证、训练和预测提交生成流程。
- `01_Data/02_DataSet_Kaggle_House/kaggle_house_pred_train.csv`：训练集，包含房屋特征和目标值 `SalePrice`。
- `01_Data/02_DataSet_Kaggle_House/kaggle_house_pred_test.csv`：测试集，包含房屋特征，不包含目标值。
- `submission.csv`：脚本生成的 Kaggle 提交文件，格式为 `Id,SalePrice`。
- `.gitignore`：忽略缓存、凭据、模型权重和临时文件。

## 模型思路

脚本中的模型为 `FeatureAttentionNet`，主要流程如下：

- 将所有训练集和测试集特征合并后统一预处理。
- 数值特征使用均值和标准差进行标准化。
- 缺失值填充为 `0`。
- 类别特征使用 one-hot 编码，并把缺失类别作为独立特征。
- 每个表格特征先映射为隐藏向量。
- 使用注意力机制学习不同特征对房价预测的重要性。
- 最后通过全连接层输出房价预测值。

评价指标使用 Kaggle 房价比赛常见的对数 RMSE，也就是对预测值和真实房价取对数后计算均方根误差。

## 环境依赖

建议使用 Python 3.10 或更新版本，并安装以下依赖：

```bash
pip install torch pandas numpy matplotlib d2l
```

如果本机有 CUDA GPU，脚本会自动使用 GPU；如果检测到多张 GPU，会使用 `DataParallel` 进行训练。

## 运行方式

在仓库根目录执行：

```bash
python kaggle_house_price_pred.py
```

脚本默认会：

1. 读取 `01_Data/02_DataSet_Kaggle_House/` 下的训练集和测试集。
2. 完成数值标准化、缺失值处理和类别特征 one-hot 编码。
3. 使用 5 折交叉验证评估模型。
4. 使用完整训练集训练最终模型。
5. 生成新的 `submission.csv`。

## 数据路径

脚本默认读取：

```text
01_Data/
  02_DataSet_Kaggle_House/
    kaggle_house_pred_train.csv
    kaggle_house_pred_test.csv
```

如果这两个文件不存在，脚本也会尝试读取下面的备用路径：

```text
01_Data/
  02_DataSet_Kaggle_House/
    kaggle_house_pred/
      train.csv
      test.csv
```

## 可调参数

当前脚本中的主要训练参数：

- `k = 5`：5 折交叉验证。
- `num_epochs = 1000`：训练轮数。
- `lr = 5`：学习率。
- `weight_decay = 0`：权重衰减。
- `batch_size = 128`：批量大小。

这些参数位于 `kaggle_house_price_pred.py` 靠近末尾的位置，可以根据训练效果调整。

## 输出结果

运行完成后会在仓库根目录生成：

```text
submission.csv
```

该文件可以直接上传到 Kaggle House Prices 比赛页面进行评分。

## 注意事项

- `submission.csv` 会被脚本重新生成，重复运行会覆盖旧结果。
- 训练过程中会绘制 RMSE 曲线；如果在无图形界面的服务器运行，可以按需要移除或注释 `plt.show()`。
- 本仓库没有提交模型权重文件，当前脚本每次运行都会重新训练模型并生成预测。
