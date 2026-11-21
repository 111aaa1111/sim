"""
run_link.py —— IM/DD 色散受限链路的系统仿真主程序

    python run_link.py

输出：out/fig1_channel.png  信道频响与谱零点
      out/fig2_ber.png      BER vs OSNR（BtB / 无均衡 / FFE / OSP）+ 净增益
      out/fig3_taps.png     抽头数–BER–插损折中
      out/report.txt       文字报告

跑通它，你的系统链路就闭环了：信道 -> 均衡 -> 探测 -> BER -> 净增益。
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

import simcore as sc

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# ============================== 参数区（改这里） ==============================
CFG = dict(
    Rs=25e9,                    # 符号率 25 GBd
    n_sps=8,                    # 每符号采样点（决定仿真带宽 = 200 GHz）
    beta=0.8,                   # 升余弦滚降（低过冲，避免削波压浅调制深度）
    half_span=8,                # 脉冲截断半长（符号）
    n_sym=2 ** 15,              # 符号数（32768）
    seed=1,
    D=17.0,                     # ps/(nm·km)，G.652 @1550 nm
    lam=1550e-9,
    L_km=20.0,                  # 传输距离
    taps_per_symbol=2,          # Δt = T / 该值；=2 即 T/2 分数间隔
    wg_loss_dB_per_ps=0.0214,   # Si 波导 3 dB/cm, ng=4.2 -> 0.0214 dB/ps
    coupling_dB=4.0,            # 两端光栅耦合 2 × 2 dB
    kappa=0.9,                  # 强度调制深度（0.9 = 大调制，链路本质非线性）
    osp_design="ls",            # "ls" = 最小二乘拟合；"chirp" = 解析解采样（反例对照）
)
TARGET_BER = 2e-2               # SD-FEC(25% OH) 门限
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
REPORT = []


def log(s=""):
    print(s)
    REPORT.append(str(s))


# ============================================================================
def _fiber_and_osp(E_tx, cfg, n_taps=None, use_osp=False):
    """一次算好光纤输出（和可选的 OSP 输出），返回 (E_rx, EL_dB)"""
    Rs, nsps = cfg["Rs"], cfg["n_sps"]
    fs = Rs * nsps
    N = len(E_tx)
    beta2 = sc.beta2_from_D(cfg["D"], cfg["lam"])
    L = cfg["L_km"] * 1e3

    H = sc.fiber_H_field(np.fft.fftfreq(N, 1 / fs), beta2, L)
    E_rx = np.fft.ifft(np.fft.fft(E_tx) * H)

    EL = 0.0
    if use_osp:
        dt = 1.0 / (Rs * cfg["taps_per_symbol"])
        stride = nsps // cfg["taps_per_symbol"]          # Δt 对应的采样点数
        bw = Rs / 2 * (1 + cfg["beta"])
        # 拟合带宽取 1.5 倍信号带宽：sqrt() 非线性带来频谱再生，场带宽宽于强度带宽
        w, EL = sc.osp_taps(n_taps, dt, beta2, L,
                            cfg["wg_loss_dB_per_ps"], cfg["coupling_dB"],
                            bandwidth=1.5 * bw, design=cfg.get("osp_design", "ls"))
        E_rx = sc.apply_fir_stride(E_rx, w, stride) * 10 ** (-EL / 20)
    return E_rx, EL


def ber_curve(cfg, sym, E_tx, osnr_list, use_osp=False, n_taps=9, use_ffe=False, seed0=100):
    E_rx, EL = _fiber_and_osp(E_tx, cfg, n_taps, use_osp)
    nsps, n_sym = cfg["n_sps"], cfg["n_sym"]
    bers, evms = [], []
    for i, o in enumerate(osnr_list):
        rng = np.random.default_rng(seed0 + i)
        y = sc.detect_with_ase(E_rx, o, nsps, rng)
        y = sc.rx_filter(y, cfg["Rs"] * nsps, cfg["Rs"])   # 接收端带宽限制
        z = sc.align_downsample(y, nsps, sym)
        if use_ffe:
            z = sc.ffe_ls(z, sym, n_taps=21)
        b, e, _ = sc.scale_and_ber(z, sym)
        bers.append(b)
        evms.append(e)
    return np.array(bers), np.array(evms), EL


def osnr_at_target(bers, osnrs, target):
    b = np.maximum(np.asarray(bers), 1e-12)
    if not np.any(b <= target):
        return None
    lb, lt = np.log10(b), np.log10(target)
    for i in range(len(b) - 1):
        if (lb[i] - lt) * (lb[i + 1] - lt) <= 0 and lb[i] != lb[i + 1]:
            t = (lt - lb[i]) / (lb[i + 1] - lb[i])
            return float(osnrs[i] + t * (osnrs[i + 1] - osnrs[i]))
    return None


# ============================================================================
def step_a_verify():
    log("=" * 80)
    log("STEP A  模型自检：仿真强度响应 vs 解析式 cos(pi·D·lam²·L·f²/c)")
    log("=" * 80)
    Rs, nsps = CFG["Rs"], CFG["n_sps"]
    fs = Rs * nsps
    N = 2 ** 17
    beta2 = sc.beta2_from_D(CFG["D"], CFG["lam"])
    L = CFG["L_km"] * 1e3

    H = sc.fiber_H_field(np.fft.fftfreq(N, 1 / fs), beta2, L)
    fr = np.fft.rfftfreq(N, 1 / fs)

    errs, eps = [], 1e-3
    for fm_target in [2e9, 5e9, 7e9, 12e9]:
        kb = int(round(fm_target * N / fs))          # 把探测频率对齐到 FFT 栅格，消除泄漏
        fm = kb * fs / N
        t = np.arange(N) / fs
        E = np.sqrt(1 + eps * np.cos(2 * np.pi * fm * t)).astype(complex)
        Id = np.abs(np.fft.ifft(np.fft.fft(E) * H)) ** 2
        spec = np.abs(np.fft.rfft(Id - Id.mean())) / N
        sim = spec[kb] / (eps / 2)
        ana = abs(sc.fiber_H_intensity(fm, CFG["D"], CFG["L_km"], CFG["lam"]))
        errs.append(abs(sim - ana) / max(ana, 1e-3))
        log(f"  f={fm/1e9:5.2f} GHz   仿真={sim:7.4f}   解析={ana:7.4f}   误差={errs[-1]*100:5.2f}%")

    f0 = sc.first_null_freq(CFG["D"], CFG["L_km"], CFG["lam"])
    bw = Rs / 2 * (1 + CFG["beta"])
    dT = sc.dispersion_spread_ps(beta2, L, bw)
    N_T = int((1 / bw + dT * 1e-12) / (1 / (Rs * CFG["taps_per_symbol"])))
    log(f"  -> 最大相对误差 {max(errs)*100:.2f}%（<2% 判定模型正确）")
    log(f"  f0 = {f0/1e9:.2f} GHz   信号带宽 = {bw/1e9:.2f} GHz   ΔT = {dT:.1f} ps")
    log(f"  抽头数估算 N_T = int[(1/B + ΔT)/Δt] = {N_T}")
    log()

    best = np.argmax(np.abs(sc.fiber_H_intensity(fr, CFG["D"], CFG["L_km"], CFG["lam"])))
    return f0, dT, N_T


def step_b_channel():
    f = np.linspace(0, 30e9, 3000)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    for L in [10, 20, 40, 80]:
        H = sc.fiber_H_intensity(f, CFG["D"], L, CFG["lam"])
        ax.plot(f / 1e9, 20 * np.log10(np.abs(H) + 1e-4), lw=1.5, label=f"{L} km")
    bw = CFG["Rs"] / 2 * (1 + CFG["beta"])
    ax.axvline(bw / 1e9, color="k", ls="--", lw=1, label=f"信号带宽 {bw/1e9:.1f} GHz")
    ax.set_xlabel("频率 (GHz)")
    ax.set_ylabel("|H(f)| (dB)")
    ax.set_title(f"平方律探测后的等效强度信道（D={CFG['D']} ps/nm/km, λ=1550 nm）")
    ax.set_ylim(-45, 5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_channel.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def step_c_ber(sym, E_tx, N_T):
    log("=" * 80)
    log("STEP C  BER vs OSNR：四条曲线 + 净增益")
    log("=" * 80)
    osnrs = np.arange(8, 42, 1.5)

    cfg_btb = dict(CFG)
    cfg_btb["L_km"] = 1e-6

    cases = []
    b, e, _ = ber_curve(cfg_btb, sym, E_tx, osnrs)
    cases.append(("BtB 无光纤", osnrs, b, 0.0))
    b, e, _ = ber_curve(CFG, sym, E_tx, osnrs)
    cases.append((f"色散 {CFG['L_km']:.0f} km 无均衡", osnrs, b, 0.0))
    b, e, _ = ber_curve(CFG, sym, E_tx, osnrs, use_ffe=True)
    cases.append(("电域 FFE (21 tap, LS)", osnrs, b, 0.0))
    b, e, EL = ber_curve(CFG, sym, E_tx, osnrs, use_osp=True, n_taps=max(9, N_T + 2))
    cases.append((f"光域 OSP ({max(9, N_T + 2)} tap, Δt=T/2)", osnrs, b, EL))

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for name, x, y, el in cases:
        z = np.flatnonzero(y <= 0)               # 首个 BER=0 之后不再画（统计上无意义）
        n = int(z[0]) if z.size else len(y)
        ax.semilogy(x[:n], y[:n], "o-", ms=3.2, lw=1.3, label=f"{name}   EL={el:.1f} dB")
    ax.axhline(TARGET_BER, color="r", ls=":", lw=1.2, label=f"SD-FEC {TARGET_BER:.0e}")
    ax.set_xlabel("接收 OSNR (dB，信号带宽内)")
    ax.set_ylabel("BER")
    ax.set_ylim(1e-6, 0.6)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title(f"25 GBd PAM4 / {CFG['L_km']:.0f} km / D={CFG['D']}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_ber.png"))
    plt.close(fig)

    log(f"  目标 BER = {TARGET_BER:.0e} 所需 OSNR：")
    req = {}
    for name, x, y, el in cases:
        r = osnr_at_target(y, x, TARGET_BER)
        req[name] = r
        log(f"    {name:30s} OSNR_req = {'N/A' if r is None else f'{r:6.2f} dB'}   EL={el:.1f} dB")
    log()

    k1, k2, k3, k4 = [c[0] for c in cases]
    if req[k1] and req[k2]:
        log(f"  色散代价（无均衡 vs BtB）  = {req[k2]-req[k1]:+6.2f} dB")
    if req[k2] and req[k4]:
        log(f"  净增益（OSP vs 无均衡）     = {req[k2]-req[k4]:+6.2f} dB  [EL 已物理作用于信号]")
    if req[k2] and req[k3]:
        log(f"  FFE 相对无均衡              = {req[k2]-req[k3]:+6.2f} dB")
    if req[k3] and req[k4]:
        log(f"  OSP 相对 FFE                = {req[k3]-req[k4]:+6.2f} dB  <- 论文要打赢的对手")

    OSNR_FIX = 32.0
    log()
    log(f"  若部分方案存在误码地板（达不到 {TARGET_BER:.0e}），改用固定 OSNR={OSNR_FIX:.0f} dB 横向对比：")
    bers = {}
    for name, x, y, el in cases:
        j = int(np.argmin(np.abs(x - OSNR_FIX)))
        bers[name] = y[j]
        log(f"    {name:30s} BER = {y[j]:9.2e}")
    if bers[k2] > 0 and bers[k4] > 0:
        log(f"  无均衡 / OSP 的 BER 改善倍数 = {np.log10(bers[k2]/max(bers[k4],1e-12)):.1f} 个数量级")
    log()
    log("  注：线性 FFE 在本场景有误码地板 —— 大调制深度下平方律检测+色散使链路本质非线性，")
    log("      这正是需要 Volterra/VNLE 或光域均衡的原因（对应设计说明第三节 IIR+FIR→二阶 Volterra）。")
    log()
    return cases, req


def step_d_taps(sym, E_tx):
    log("=" * 80)
    log("STEP D  抽头数扫描：性能与插损的折中")
    log("=" * 80)
    taps = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    osnr_fix = 32.0
    rows = []
    for nt in taps:
        b, e, EL = ber_curve(CFG, sym, E_tx, [osnr_fix], use_osp=True, n_taps=nt)
        rows.append((nt, EL, b[0], e[0]))
        log(f"  N={nt:3d}   EL={EL:5.2f} dB   BER={b[0]:9.2e}   EVM={e[0]*100:5.1f}%")
    log()

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    nt = [r[0] for r in rows]
    ax1.semilogy(nt, [max(r[2], 1e-7) for r in rows], "o-", color="#c0392b", lw=1.6, label="BER")
    ax1.set_xlabel("OSP 抽头数 N")
    ax1.set_ylabel("BER", color="#c0392b")
    ax1.grid(alpha=0.3, which="both")
    ax2 = ax1.twinx()
    ax2.plot(nt, [r[1] for r in rows], "s--", color="#2c3e50", lw=1.2)
    ax2.set_ylabel("插入损耗 EL (dB)", color="#2c3e50")
    ax1.set_title(f"抽头数 → 性能 → 插损（OSNR 固定 {osnr_fix:.0f} dB）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_taps.png"))
    plt.close(fig)
    return rows


# ============================================================================
def main():
    t0 = time.time()
    log(f"配置  {CFG['Rs']/1e9:.0f} GBd PAM4 | {CFG['L_km']:.0f} km | D={CFG['D']} "
        f"| Δt=T/{CFG['taps_per_symbol']} | {CFG['n_sym']} 符号")
    log()
    f0, dT, N_T = step_a_verify()

    sym, I, E = sc.make_waveform(CFG["n_sym"], CFG["n_sps"], CFG["beta"],
                                 CFG["half_span"], CFG["seed"], kappa=CFG["kappa"])

    log(f"[图] {step_b_channel()}")
    log()
    step_c_ber(sym, E, N_T)
    log(f"[图] {os.path.join(OUT, 'fig2_ber.png')}")
    log()
    step_d_taps(sym, E)
    log(f"[图] {os.path.join(OUT, 'fig3_taps.png')}")
    log()
    log(f"总耗时 {time.time()-t0:.1f} s")

    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(REPORT))
    print(f"报告 -> {os.path.join(OUT, 'report.txt')}")


if __name__ == "__main__":
    main()
