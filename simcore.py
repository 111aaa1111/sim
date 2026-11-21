"""
simcore.py —— IM/DD 色散均衡系统仿真的核心模型（零商业软件依赖）

物理链条：
    归一化 PAM4 符号 -> 升余弦脉冲成型 -> 强度调制 -> 复光场 sqrt(I)
    -> 光纤色散（复基带相位滤波 H=exp(-j beta2 L w^2/2)）
    -> [OSP：光域复抽头 FIR]  ->  ASE -> 平方律探测 |.|^2
    -> 符号对齐 -> AGC 标定 -> 判决 -> BER

归一化约定：平均光功率 = 1，时间 s，频率 Hz。
"""

import numpy as np

C0 = 299792458.0
PAM4_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])
GRAY = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])   # 索引 0,1,2,3 -> 00,01,11,10


# ---------------------------------------------------------------- 光纤色散
def beta2_from_D(D_ps_nm_km, lam_m):
    """D [ps/(nm·km)] -> beta2 [s^2/m]"""
    return -(D_ps_nm_km * 1e-6) * lam_m ** 2 / (2 * np.pi * C0)


def fiber_H_field(f, beta2, L):
    """复光场传递函数"""
    w = 2 * np.pi * f
    return np.exp(-1j * beta2 * L * w ** 2 / 2)


def fiber_H_intensity(f, D_ps_nm_km, L_km, lam_m):
    """平方律探测后的等效强度信道（小信号）cos(pi D lam^2 L f^2 / c)，D 内部换算到 s/m^2"""
    D_si = D_ps_nm_km * 1e-6
    return np.cos(np.pi * D_si * lam_m ** 2 * (L_km * 1e3) * f ** 2 / C0)


def first_null_freq(D_ps_nm_km, L_km, lam_m):
    return np.sqrt(C0 / (2 * D_ps_nm_km * 1e-6 * lam_m ** 2 * (L_km * 1e3)))


def dispersion_spread_ps(beta2, L, bandwidth_hz):
    return abs(beta2 * L * 2 * np.pi * bandwidth_hz) * 1e12


# ---------------------------------------------------------------- 信号生成
def rc_pulse(beta, half_span_sym, n_sps):
    """升余弦脉冲，符号率采样点处零 ISI；归一化为峰值 = 1（符号幅度直接可读）"""
    t = np.arange(-half_span_sym * n_sps, half_span_sym * n_sps + 1) / n_sps
    num = np.cos(np.pi * beta * t)
    den = 1 - (2 * beta * t) ** 2
    h = np.sinc(t) * num / den
    sing = np.abs(den) < 1e-8
    if np.any(sing):
        h[sing] = np.pi / 4 * np.sinc(1 / (2 * beta))
    return h / h[h.size // 2]           # 峰值归一化：符号中心幅度 = 1


def make_waveform(n_sym, n_sps, beta, half_span_sym, seed, kappa=0.45):
    """返回 (符号索引, 强度波形 I, 复光场 E)"""
    rng = np.random.default_rng(seed)
    sym = rng.integers(0, 4, n_sym)

    up = np.zeros(n_sym * n_sps)
    up[::n_sps] = PAM4_LEVELS[sym]
    h = rc_pulse(beta, half_span_sym, n_sps)
    full = np.convolve(up, h, mode="full")
    off = (len(h) - 1) // 2
    w = full[off:off + n_sym * n_sps]

    # 升余弦（非根升余弦）存在码型相关过冲；按最坏过冲归一化并加直流偏置，
    # 等价于把调制器偏置在保证不削波的工作点上（这是真实链路必须做的事）
    OVERSHOOT = 1.25
    u = w / (3.0 * OVERSHOOT)              # 符号中心电平：±0.80 / ±0.267
    I = 1.0 + kappa * u
    if I.min() <= 0.02:
        raise ValueError(f"强度波形出现非正值 min(I)={I.min():.4f}，请降低 kappa 或提高 OVERSHOOT")
    return sym, I, np.sqrt(I).astype(complex)


# ---------------------------------------------------------------- OSP 模型
def osp_taps(n_taps, dt, beta2, L, wg_loss_dB_per_ps, coupling_dB=0.0,
             bandwidth=None, design="ls", window=None):
    """
    设计 OSP 复抽头。返回 (w, EL_dB)。

    design="ls"    : 在信号带内最小二乘拟合目标相频响应 exp(+j*beta2*L*w^2/2)
                     —— 这是正确做法，9 抽头即可把残余幅度压到 0.999~1.001
    design="chirp" : 直接对解析逆滤波器冲激响应 exp(+j t^2/(2|beta2|L)) 采样
                     —— 会带来 ±5 dB 截断纹波，留作反例对照

    归一化：频响峰值 = 1（真全通相位均衡器，不引入虚假增益）；
    插入损耗 EL = 平均波导损耗 + 耦合损耗，作为标量衰减作用于信号。
    """
    if n_taps % 2 == 0:
        n_taps += 1
    k = np.arange(n_taps) - n_taps // 2
    t = k * dt

    if design == "chirp":
        w = np.exp(1j * (-t ** 2 / (2 * beta2 * L))).astype(complex)
        if window == "gaussian" and n_taps > 2:
            w *= np.exp(-0.5 * (t / (np.abs(t).max())) ** 2 * 2.0)
        fgrid = np.linspace(-0.5 / dt, 0.5 / dt, 4001)
        Hpk = np.abs(np.exp(-1j * 2 * np.pi * np.outer(fgrid, t)) @ w).max()
        w /= Hpk
    else:
        if bandwidth is None:
            raise ValueError("design='ls' 需要传入 bandwidth")
        f = np.linspace(-bandwidth / 2, bandwidth / 2, 1201)
        Ht = np.exp(1j * beta2 * L * (2 * np.pi * f) ** 2 / 2)
        A = np.exp(-1j * 2 * np.pi * np.outer(f, t))
        w, *_ = np.linalg.lstsq(A, Ht, rcond=None)
        w /= np.abs(A @ w).max()                    # 带内峰值归一化

    delay_ps = np.abs(k) * dt * 1e12
    EL = float(np.mean(wg_loss_dB_per_ps * delay_ps)) + coupling_dB
    return w, EL


def rx_filter(y, fs, Rs, order=4, ratio=0.75):
    """接收端带宽限制（4 阶 Butterworth 零相位），这是真实接收机必有的环节"""
    from scipy.signal import butter, filtfilt
    fc = ratio * Rs
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, y, padlen=min(3 * max(len(a), len(b)), len(y) - 1))


def apply_fir_stride(x, taps, stride):
    """y[n] = sum_i w_i * x[n - (i - N//2)*stride]"""
    n, y = len(x), np.zeros_like(x)
    off = len(taps) // 2
    for i, w in enumerate(taps):
        if w == 0:
            continue
        s = (i - off) * stride
        if s >= 0:
            y[s:] += w * x[:n - s]
        else:
            y[:s] += w * x[-s:]
    return y


# ---------------------------------------------------------------- 探测与判决
def detect_with_ase(E, osnr_db, n_sps, rng):
    """复光场叠加 ASE 后平方律探测。OSNR = 信号带宽内信号/噪声功率比。"""
    sigma2 = 10 ** (-osnr_db / 10) * n_sps      # 噪声摊到采样带宽 fs = n_sps*Rs
    n = np.sqrt(sigma2 / 2) * (rng.standard_normal(len(E)) + 1j * rng.standard_normal(len(E)))
    return np.abs(E + n) ** 2


def align_downsample(y, n_sps, sym, coarse=4096, max_lag=8):
    """
    数据辅助同步：在 [0, n_sps) 内搜抽样相位、在 ±max_lag 符号内搜整数时延，
    判据为与已知训练符号的归一化 MSE 最小（而不是"方差最大"——那是错判据）。
    返回已对齐、已隔直、已归一化的符号率序列 z。
    """
    tgt = PAM4_LEVELS[sym] / 3.0
    n = len(sym)
    k = n // 2
    m = min(coarse, n)
    best_z, best_err = None, np.inf

    for s in range(n_sps):
        z = y[s::n_sps]
        if len(z) < n:
            continue
        z = z[:n].astype(float)
        zc = z - z.mean()

        c = np.correlate(zc[:m], tgt[:m], "full")
        d = int(np.argmax(np.abs(c)) - (m - 1))
        if abs(d) > max_lag:
            d = 0
        zz = np.roll(zc, -d)

        g = np.dot(zz[:k], tgt[:k]) / (np.dot(zz[:k], zz[:k]) + 1e-30)
        err = np.mean((g * zz[:k] - tgt[:k]) ** 2)
        if err < best_err:
            best_err, best_z = err, zz
    return best_z


def scale_and_ber(z, sym, train_frac=0.5):
    """
    训练段做隔直 + 增益标定（AGC），测试段判决计 BER。训练/测试严格分离。
    目标电平归一化为 ±1, ±1/3（即 PAM4 电平 /3），判决门限 ±2/3 与 0。
    """
    n = len(z)
    k = int(n * train_frac)
    tgt = PAM4_LEVELS[sym] / 3.0

    zc = z - z[:k].mean()                                   # 隔直：去掉 PAM4 的偏置
    g = np.dot(zc[:k], tgt[:k]) / (np.dot(zc[:k], zc[:k]) + 1e-30)
    zs = g * zc

    idx = np.searchsorted(np.array([-2 / 3, 0.0, 2 / 3]), zs)
    ber = np.mean(GRAY[idx][k:] != GRAY[sym][k:])
    evm = np.sqrt(np.mean((zs[k:] - tgt[k:]) ** 2) / np.mean(tgt[k:] ** 2))
    return float(ber), float(evm), zs


# ---------------------------------------------------------------- 电域 FFE
def ffe_ls(z, sym, n_taps=21, train_frac=0.5):
    """T 间隔 FIR，最小二乘训练（DSP 基线）。输入端先隔直。"""
    n = len(z)
    k = int(n * train_frac)
    tgt = PAM4_LEVELS[sym] / 3.0

    zc = z - z.mean()
    X = np.stack([np.roll(zc, j) for j in range(n_taps)], axis=1)[:k]
    w, *_ = np.linalg.lstsq(X, tgt[:k], rcond=None)

    y = np.zeros(n)
    for j in range(n_taps):
        y += w[j] * np.roll(zc, j)
    return y
