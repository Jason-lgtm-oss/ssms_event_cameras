"""
S5SSM Forward Method - 完全注释版本

本文件提供了 S5SSM.forward() 方法的完全逐行注释版本
用于深度理解张量形状变化和计算流程
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

# ═══════════════════════════════════════════════════════════════════════════════
# 部分 1: 核心辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def as_complex(t: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
    """
    将形状为 (..., 2) 的实数对张量转换为复数张量
    
    Args:
        t: 形状为 (..., 2) 的张量,其中 [..., 0] 是实部,​​​[..., 1] 是虚部
        dtype: 目标复数类型 (default: torch.complex64)
    
    Returns:
        复数张量,形状 (...)
    
    例子:
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 形状 (2, 2)
        z = as_complex(t)  # 形状 (2,), 值为 [1+2j, 3+4j]
    """
    assert t.shape[-1] == 2, "as_complex 只能处理最后维度为 2 的张量"
    # 使用 torch.complex 构造复数张量
    # 从 (..., 2) → (...)
    nt = torch.complex(t[..., 0], t[..., 1])
    
    # 确保数据类型一致
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    
    return nt


def binary_operator(
    q_i: Tuple[torch.Tensor, torch.Tensor],
    q_j: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    关联扫描的二元运算符
    
    用于计算线性递推关系: x_t = A_t @ x_{t-1} + b_t
    
    数学运算:
        q_i = (A_i, b_i)  其中 A_i 是标量,b_i 是向量
        q_j = (A_j, b_j)
        
        返回: (A_j * A_i, A_j * b_i + b_j)
        
        这代表: x = A_i @ x_prev + b_i
               x' = A_j @ x + b_j
                  = A_j @ (A_i @ x_prev + b_i) + b_j
                  = (A_j * A_i) @ x_prev + (A_j * b_i + b_j)
    
    关键性质: 满足结合律
        (q_a ⊕ q_b) ⊕ q_c = q_a ⊕ (q_b ⊕ q_c)
        这允许树形并行扫描
    
    Args:
        q_i: (A_i, b_i) 其中 A_i: 标量 (复数), b_i: (P,) 向量
        q_j: (A_j, b_j) 其中 A_j: 标量 (复数), b_j: (P,) 向量
    
    Returns:
        (A_out, b_out) 新的元组
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    
    # torch.addcmul: add(b_j, A_j * b_i) = b_j + A_j * b_i
    return A_j * A_i, torch.addcmul(b_j, A_j, b_i)


def discretize_zoh(
    Lambda: torch.Tensor,    # (P, 2) - 复数特征值
    B_tilde: torch.Tensor,   # (P, h) - 复数输入矩阵
    Delta: torch.Tensor      # (P,) 或 (L, P) - 离散化步长
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-Order Hold (ZOH) 离散化
    
    连续系统: dx/dt = Λ @ x + B @ u
    
    在时间区间 [t, t+Δ] 上,假设 u 恒定 (zero-order hold):
    
    x(t+Δ) = e^{Λ*Δ} @ x(t) + ∫_0^Δ e^{Λ*τ} dτ @ B @ u(t)
    
    定义:
        Λ_bar = e^{Λ*Δ}
        B_bar = Λ^{-1} @ (e^{Λ*Δ} - I) @ B
              = (1/Λ) * (e^{Λ*Δ} - 1) @ B
    
    离散系统: x_{t+1} = Λ_bar @ x_t + B_bar @ u_t
    
    Args:
        Lambda: (P, 2) 张量,存储为 [real, imag] 对
                通过 torch.view_as_complex 转换后为 (P,) 复数
        B_tilde: (P, h) 复数张量 (已经是 V^{-1} @ B)
        Delta: (P,) 或 (L, P) 离散化步长
    
    Returns:
        Lambda_bar: 同 Lambda 的形状,离散化的特征值矩阵
        B_bar: 同 B_tilde 的形状,离散化的输入矩阵
    """
    # 1. 转换 Lambda 为复数
    Lambda = torch.view_as_complex(Lambda)  # (P,) 复数
    
    # 2. 计算 Λ_bar = e^{Λ*Δ}
    # 当 Delta 是 (P,) 时: Λ*Δ 是 element-wise 乘法 (P,)
    # 当 Delta 是 (L, P) 时: Λ*Δ 广播到 (L, P)
    Lambda_bar = torch.exp(Lambda * Delta)  # 形状同 Lambda * Delta
    
    # 3. 计算 B_bar = (1/Λ) * (Λ_bar - 1) * B_tilde
    # (1/Λ * (Λ_bar - 1))[..., None] 添加维度以与 B_tilde 兼容
    # [..., None]: (P,) → (P, 1) 或 (L, P, 1) → (L, P, 1)
    # 然后与 B_tilde 广播乘法
    B_bar = (1 / Lambda * (Lambda_bar - 1))[..., None] * B_tilde
    
    # 4. 转换回实数对形式以匹配 forward 方法的期望
    Lambda_bar = torch.view_as_real(Lambda_bar)  # (..., 2) 形式
    B_bar = torch.view_as_real(B_bar)            # (..., 2) 形式
    
    return Lambda_bar, B_bar


def discretize_bilinear(
    Lambda: torch.Tensor,
    B_tilde: torch.Tensor,
    Delta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    双线性变换 (Bilinear Transform) 离散化
    
    另一种离散化方法,常用于信号处理
    
    公式:
        BL = (I - Δ/2 * Λ)^{-1}
        Λ_bar = BL @ (I + Δ/2 * Λ)
        B_bar = BL @ Δ @ B
    
    Args:
        Lambda: (P, 2) 张量
        B_tilde: (P, h) 复数张量
        Delta: (P,) 或 (L, P) 离散化步长
    
    Returns:
        Lambda_bar, B_bar (离散化参数)
    """
    # 1. 转换为复数
    Lambda = torch.view_as_complex(Lambda)  # (P,)
    
    # 2. 创建恒等矩阵元素 (对角矩阵的对角线)
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    
    # 3. 计算 BL = (I - Δ/2 * Λ)^{-1}
    # 对于对角矩阵,求逆就是逐元素倒数
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)  # (P,) 或 (L, P)
    
    # 4. 计算 Λ_bar
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    
    # 5. 计算 B_bar
    B_bar = (BL * Delta)[..., None] * B_tilde
    
    return Lambda_bar, B_bar


# ═══════════════════════════════════════════════════════════════════════════════
# 部分 2: Apply SSM 函数 (核心计算)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ssm(
    Lambda_bars: torch.Tensor,    # (P, 2) 或 (L, P, 2)
    B_bars: torch.Tensor,          # (P, h, 2) 或 (L, P, h, 2)
    C_tilde: torch.Tensor,         # (h, P, 2)
    D: torch.Tensor,               # (h,)
    input_sequence: torch.Tensor,  # (L, h)
    prev_state: torch.Tensor,      # (P,)
    bidir: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用 SSM 使用并行扫描
    
    计算线性递推系统:
        h_t = Λ_bar_t @ h_{t-1} + B_bar_t @ u_t
        y_t = (C_tilde @ h_t).real + D * u_t
    
    关键思想:
        1. 将离散时间系统表示为关联运算的序列
        2. 使用并行扫描高效计算所有时刻的状态
        3. 时间复杂度: O(log L) (深度) vs O(L) (顺序)
    
    Args:
        Lambda_bars: 离散化的对角特征值矩阵
        B_bars: 离散化的输入矩阵
        C_tilde: 输出投影矩阵 (已在特征空间中)
        D: 直通矩阵
        input_sequence: 输入信号 (L, h)
        prev_state: 前一时刻的隐状态 (P,)
        bidir: 是否使用双向扫描
    
    Returns:
        output: (L, h) 输出序列
        final_state: (P,) 最后一个隐状态
    """
    
    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ 第 1 步: 转换张量为复数表示                                       ║
    # ╚════════════════════════════════════════════════════════════════════╝
    
    # 所有计算都在复数域进行 (符号需求)
    B_bars = as_complex(B_bars)              # (P, h) 或 (L, P, h)
    C_tilde = as_complex(C_tilde)            # (h, P)
    Lambda_bars = as_complex(Lambda_bars)    # (P,) 或 (L, P)
    
    # 输入也转换为复数 (为了与 Lambda_bars 的类型一致)
    cinput_sequence = input_sequence.type(Lambda_bars.dtype)  # (L, h)
    
    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ 第 2 步: 计算 Bu 元素 (所有时刻的 B @ u)                         ║
    # ╚════════════════════════════════════════════════════════════════════╝
    
    # torch.vmap: 对输入序列的每个时刻应用同一个函数
    # vmap 自动处理 map 操作,避免显式循环
    
    if B_bars.ndim == 3:
        # 情况 1: 动态时间步长 (每个时刻的 B 不同)
        # B_bars 形状: (L, P, h)
        # cinput_sequence 形状: (L, h)
        # vmap 在第一个维度 (时间维度) 上进行映射
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(
            B_bars, cinput_sequence
        )
        # 输出 Bu_elements: (L, P)
        # 每个时刻: B_bar[t] @ u[t] → (P,)
    else:
        # 情况 2: 静态时间步长 (所有时刻的 B 相同) - 更常见
        # B_bars 形状: (P, h)
        # cinput_sequence 形状: (L, h)
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)
        # 输出 Bu_elements: (L, P)
        # 每个时刻: B_bars @ u[t] → (P,)
    
    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ 第 3 步: 处理 Lambda_bars 的维度                                   ║
    # ╚════════════════════════════════════════════════════════════════════╝
    
    if Lambda_bars.ndim == 1:
        # Lambda_bars 是 1D: (P,)
        # 需要在时间维度上复制,以匹配并行扫描的要求
        # 复制到 (L, P),每行相同
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)
        # 输出 Lambda_bars: (L, P)
    
    # 现在 Lambda_bars 和 Bu_elements 都是 (L, P) 形状
    
    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ 第 4 步: 初始化前缀扫描 (处理前一状态)                            ║
    # ╚════════════════════════════════════════════════════════════════════╝
    
    # 关键步骤!
    # 在并行扫描中,第一个元素被视为初始值
    # h_0 = Λ_bar[0] @ h_{-1} + B_bar[0] @ u_0
    # 
    # 我们通过修改 Λ_bar[0] 来直接将 h_{-1} 编码到扫描中:
    # 修改后的 Λ_bar[0] 将与一个虚拟的 "0 状态" 相乘
    # 得到 Λ_bar[0] * prev_state (直接编码前一状态)
    
    Lambda_bars[0] = Lambda_bars[0] * prev_state
    # Lambda_bars[0]: (P,) * (P,) → (P,) element-wise 乘法
    
    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ 第 5 步: 关联扫描 (核心并行计算)                                   ║
    # ╚════════════════════════════════════════════════════════════════════╝
    
    # 执行关联扫描来计算所有时刻的状态
    # 这是一个树形算法,深度为 O(log L)
    
    _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))
    
    # 说明:
    #   associative_scan(op, (a1, b1), (a2, b2), ...)
    #   计算前缀序列:
    #     (a1, b1)
    #     (a1, b1) ⊕ (a2, b2)
    #     ((a1, b1) ⊕ (a2, b2)) ⊕ (a3, b3)
    #     ...
    #   其中 ⊕ 是 binary_operator
    #
    # 返回值:
    #   _: 最终的累积 A 值 (unused)
    #   xs: 所有时刻的 b 值序列 (即隐状态)
    
    # 输出 xs: (L, P) 复数
    # xs[t] 是第 t 时刻的隐状态
    
    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ 第 6 步: 双向处理 (可选)                                          ║
    # ╚════════════════════════════════════════════════════════════════════╝
    
    if bidir:
        # 反向运行关联扫描
        _, xs2 = associative_scan(
            binary_operator, (Lambda_bars, Bu_elements), reverse=True
        )
        # xs2 也是 (L, P)
        
        # 拼接前向和后向的隐状态
        xs = torch.cat((xs, xs2), axis=-1)
        # 输出 xs: (L, 2P) - 双向隐状态
    
    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ 第 7 步: 计算直接路径 D @ u                                        ║
    # ╚════════════════════════════════════════════════════════════════════╝
    
    # D 是 (h,) 的向量
    # 对每个时刻的输入应用:
    Du = torch.vmap(lambda u: D * u)(input_sequence)
    # 输出 Du: (L, h)
    # 每个时刻: D * u[t] → (h,) element-wise 乘法
    
    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ 第 8 步: 计算最终输出                                              ║
    # ╚════════════════════════════════════════════════════════════════════╝
    
    # 完整的输出公式: y_t = (C_tilde @ x_t).real + D * u_t
    
    # 首先计算 C_tilde @ x_t
    # C_tilde: (h, P) 复数
    # xs[t]: (P,) 复数
    # C_tilde @ xs[t]: (h,) 复数
    
    y_states = torch.vmap(lambda x: (C_tilde @ x))(xs)
    # 输出 y_states: (L, h) 复数
    
    # 提取实部 (虚部应该接近零,由于我们的设计)
    y_real = y_states.real  # (L, h) 实数
    
    # 添加直接路径
    output = y_real + Du  # (L, h) + (L, h) → (L, h)
    
    # 提取最终隐状态 (最后一个时刻)
    final_state = xs[-1].real  # (P,) 实数
    
    return output, final_state


# ═══════════════════════════════════════════════════════════════════════════════
# 部分 3: S5SSM Forward 方法
# ═══════════════════════════════════════════════════════════════════════════════

class S5SSM_Annotated(nn.Module):
    """
    S5 State Space Model - 带详细注释的版本
    
    这是一个线性状态空间模型,使用对角化的 HiPPO 矩阵作为状态转移矩阵
    """
    
    def __init__(
        self,
        lambdaInit: torch.Tensor,          # (P,) 复数 - 初始特征值
        h: int,                             # 输入特征维度
        p: int,                             # 状态维度
        dt_min: float = 0.001,              # 时间步长最小值
        dt_max: float = 0.1,                # 时间步长最大值
        liquid: bool = False,               # 是否使用 Liquid SSM
        discretization: str = "bilinear",   # 离散化方法
    ):
        super().__init__()
        
        # 将复数特征值存储为实数对 (复数被表示为 [..., 2])
        self.Lambda = nn.Parameter(torch.view_as_real(lambdaInit))
        # 形状: (P, 2) - 每行是一个复数的 [实部, 虚部]
        
        # 输入矩阵 (在特征空间中)
        self.B = nn.Parameter(torch.randn(p, h) * 0.1)  # (P, h)
        
        # 输出矩阵 (在特征空间中)
        self.C = nn.Parameter(torch.view_as_real(torch.randn(h, p) * 0.1))  # (h, P, 2)
        
        # 直通矩阵
        self.D = nn.Parameter(torch.rand(h))  # (h,)
        
        # 学习的时间步长
        import numpy as np
        log_steps = np.random.uniform(np.log(dt_min), np.log(dt_max), p)
        self.log_step = nn.Parameter(torch.tensor(log_steps, dtype=torch.float32))
        # 形状: (P,)
        
        # 选择离散化方法
        if discretization == "bilinear":
            self.discretize = discretize_bilinear
        else:
            self.discretize = discretize_zoh
    
    def get_BC_tilde(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取变换后的 B 和 C 矩阵 (在特征空间中)"""
        B_tilde = as_complex(self.B)      # (P, h) 复数
        C_tilde = as_complex(self.C)      # (h, P) 复数
        return B_tilde, C_tilde
    
    def forward(
        self,
        signal: torch.Tensor,             # (L, h) 输入信号
        prev_state: torch.Tensor,         # (P,) 前一状态
        step_scale: float = 1.0            # 步长缩放
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        S5SSM 的前向传播
        
        计算流程:
            1. 获取变换的参数 B_tilde, C_tilde
            2. 计算离散化步长 Δ
            3. 离散化参数 Λ 和 B
            4. 应用 SSM (使用并行扫描)
            5. 返回输出和最终状态
        
        Args:
            signal: (L, h) 输入序列
            prev_state: (P,) 前一时刻的隐状态
            step_scale: 步长缩放因子
        
        Returns:
            output: (L, h) 输出序列
            final_state: (P,) 最后一个隐状态
        """
        
        # ┌─ 第1步: 获取变换的参数 ─────────────────────────────────────────┐
        
        B_tilde, C_tilde = self.get_BC_tilde()
        # B_tilde:   (P, h) 复数
        # C_tilde:   (h, P) 复数
        
        # └─ 第1步 完成 ─────────────────────────────────────────────────────┘
        
        # ┌─ 第2步: 计算离散化步长 ───────────────────────────────────────────┐
        
        # 公式: Δ_i = step_scale * exp(log_step_i)
        # 这确保 Δ > 0
        
        step = step_scale * torch.exp(self.log_step)
        # step: (P,) - 每个状态维度的步长
        
        # 如果 step_scale 是张量 (时间变化):
        if torch.is_tensor(step_scale) and step_scale.ndim > 0:
            # step_scale: (L,)
            # 计算 Δ[t, i] = step_scale[t] * exp(log_step[i])
            step = step_scale[:, None] * torch.exp(self.log_step)
            # step: (L, P) - 时间和状态维度都变化
        
        # └─ 第2步 完成 ─────────────────────────────────────────────────────┘
        
        # ┌─ 第3步: 参数离散化 ───────────────────────────────────────────────┐
        
        # 将连续时间参数转换为离散时间参数
        # 使用选定的离散化方法 (ZOH 或 Bilinear)
        
        Lambda_bars, B_bars = self.discretize(self.Lambda, B_tilde, step)
        
        # Lambda_bars:
        #   输入: (P, 2)
        #   输出: (P, 2) 或 (L, P, 2) - 取决于 step 的维度
        #   意义: 离散化的对角状态矩阵
        
        # B_bars:
        #   输入: (P, h)
        #   输出: (P, h, 2) 或 (L, P, h, 2)
        #   意义: 离散化的输入矩阵
        
        # └─ 第3步 完成 ─────────────────────────────────────────────────────┘
        
        # ┌─ 第4步: 应用 SSM ─────────────────────────────────────────────────┐
        
        # 调用核心 SSM 应用函数
        # 这进行关联扫描来计算所有时刻的状态
        
        output, final_state = apply_ssm(
            Lambda_bars,      # 离散化的特征值
            B_bars,           # 离散化的输入矩阵
            C_tilde,          # 输出矩阵 (已在特征空间)
            self.D,           # 直通矩阵
            signal,           # 输入信号
            prev_state,       # 前一状态
            bidir=False       # 单向模式
        )
        
        # output: (L, h) - 输出序列
        # final_state: (P,) - 最后时刻的隐状态
        
        # └─ 第4步 完成 ─────────────────────────────────────────────────────┘
        
        return output, final_state


# ═══════════════════════════════════════════════════════════════════════════════
# 部分 4: 数值例子和验证
# ═══════════════════════════════════════════════════════════════════════════════

def example_forward_pass():
    """
    完整的数值例子
    """
    
    print("=" * 80)
    print("S5SSM Forward Pass 数值例子")
    print("=" * 80)
    
    # 设置参数
    h = 4           # 输入特征维度
    P = 8           # 状态维度
    L = 3           # 序列长度
    
    print(f"\n参数设置:")
    print(f"  h (输入维度): {h}")
    print(f"  P (状态维度): {P}")
    print(f"  L (序列长度): {L}")
    
    # 创建模型
    Lambda_init = torch.randn(P) * 0.1 - 0.5j  # 初始化为负实部 (稳定)
    model = S5SSM_Annotated(
        lambdaInit=Lambda_init,
        h=h,
        p=P,
        dt_min=0.001,
        dt_max=0.1,
        discretization="bilinear"
    )
    
    # 创建输入
    signal = torch.randn(L, h)
    prev_state = torch.zeros(P)
    
    print(f"\n输入张量形状:")
    print(f"  signal:     {signal.shape}")
    print(f"  prev_state: {prev_state.shape}")
    
    # 前向传播
    output, final_state = model(signal, prev_state)
    
    print(f"\n输出张量形状:")
    print(f"  output:      {output.shape}")
    print(f"  final_state: {final_state.shape}")
    
    print(f"\n张量统计:")
    print(f"  signal:     mean={signal.mean():.4f}, std={signal.std():.4f}")
    print(f"  output:     mean={output.mean():.4f}, std={output.std():.4f}")
    print(f"  final_state: mean={final_state.mean():.4f}, std={final_state.std():.4f}")
    
    # 检查梯度
    loss = output.sum()
    loss.backward()
    
    print(f"\n梯度检查:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name:15} grad_mean={param.grad.mean():.6f}, grad_std={param.grad.std():.6f}")


def example_step_by_step():
    """
    逐步追踪每个计算步骤
    """
    
    print("\n" + "=" * 80)
    print("逐步追踪计算流程")
    print("=" * 80)
    
    h, P, L = 2, 3, 2
    
    # 简化的参数
    Lambda = torch.tensor([[-1.0 + 1.0j, -2.0 + 0.5j, -0.5 - 0.5j]], dtype=torch.complex64).T
    B = torch.randn(P, h, dtype=torch.complex64) * 0.1
    C = torch.randn(h, P, dtype=torch.complex64) * 0.1
    D = torch.rand(h)
    log_step = torch.tensor([-6.9, -6.0, -5.0])  # log of [0.001, 0.0025, 0.0067]
    
    signal = torch.randn(L, h)
    prev_state = torch.randn(P)
    
    print(f"\n初始参数:")
    print(f"  Lambda shape: {Lambda.shape}")
    print(f"  B shape: {B.shape}")
    print(f"  C shape: {C.shape}")
    print(f"  D shape: {D.shape}")
    
    # 步骤 1: 计算步长
    print(f"\n步骤 1: 计算步长")
    step = torch.exp(log_step)
    print(f"  step = exp(log_step) = {step}")
    
    # 步骤 2: 离散化
    print(f"\n步骤 2: 参数离散化 (Bilinear)")
    Identity = torch.ones_like(Lambda)
    BL = 1 / (Identity - (step[:, None] * Lambda))
    Lambda_bar = BL * (Identity + (step[:, None] * Lambda))
    B_bar = (BL * step[:, None]) * B[:, :]
    
    print(f"  Lambda_bar[0] = {Lambda_bar[0]}")
    print(f"  B_bar[0] = {B_bar[0]}")
    
    # 步骤 3: 计算 Bu
    print(f"\n步骤 3: 计算 Bu 元素")
    Bu = B_bar @ signal.T  # (P, h) @ (h, L) → (P, L)
    print(f"  Bu shape: {Bu.shape}")
    print(f"  Bu[0, :] = {Bu[0, :]}")
    
    # 步骤 4: 顺序扫描 (用于验证)
    print(f"\n步骤 4: 顺序扫描 (验证)")
    h_t = prev_state.clone()
    for t in range(L):
        h_t = Lambda_bar[:, t] * h_t + Bu[:, t]
        print(f"  h[{t}] = {h_t}")
    
    # 步骤 5: 输出
    print(f"\n步骤 5: 计算输出")
    y_t = (C @ h_t).real + D
    print(f"  y[last] = {y_t}")


if __name__ == "__main__":
    # 运行例子
    example_forward_pass()
    example_step_by_step()
