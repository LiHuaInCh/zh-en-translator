"""全局配置，所有脚本共用。"""
from pathlib import Path

ROOT = Path(__file__).parent

# 模型路径 (本地 models/ 目录)
MODEL_ZH_EN = str(ROOT / "models" / "zh-en-final")    # 中→英
MODEL_EN_ZH = str(ROOT / "models" / "en-zh-final")    # 英→中

# 数据
MAX_SEQ_LENGTH = 128                  # 句子最大 token 数

# 训练
BATCH_SIZE = 16                       # RTX 3070 8GB 适合 16-32
EVAL_BATCH_SIZE = 32                 # 评估时可用更大 batch
GRADIENT_ACCUMULATION = 2             # 梯度累积 = 等效 batch_size * 2
LEARNING_RATE = 2e-5                     # 更低学习率防止模型崩坏
NUM_EPOCHS = 3                         # 减少轮数防过拟合
WARMUP_STEPS = 500
LOGGING_STEPS = 100
EVAL_STEPS = 2000                    # 降低评估频率加速训练
SAVE_STEPS = 4000                    # 保存 checkpoint 频率

# 保存路径
OUTPUT_DIR_ZH_EN = ROOT / "models" / "zh-en-final"
OUTPUT_DIR_EN_ZH = ROOT / "models" / "en-zh-final"
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
