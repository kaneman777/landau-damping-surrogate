import torch
import torch.nn as nn

class LandauSurrogate(nn.Module):
    def __init__(self):
        super(LandauSurrogate, self).__init__()
        # 入力: [Te, Lx, t] の3次元
        # 出力: [Log10_Energy] の1次元
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),         # 滑らかな非線形性を導入
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)   # 最終的なエネルギー（対数）を出力
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # 動作確認: 適当な入力を入れてエラーが出ないかチェック
    model = LandauSurrogate()
    test_input = torch.randn(5, 3) # 5データ分、3入力
    test_output = model(test_input)
    print(f"Model structure validated. Output shape: {test_output.shape}")