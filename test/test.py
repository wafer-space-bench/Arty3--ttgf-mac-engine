# SPDX-License-Identifier: Apache-2.0
"""
Cocotb test suite for tt_um_arty3_mac_engine (ttgf-tiny-conv).

Drives the DUT through its SPI-mode-0 slave on uio[3:0]:
    uio[0] = CS_N (in)
    uio[1] = SCLK (in)
    uio[2] = MOSI (in)
    uio[3] = MISO (out, sampled from dut.uio_out)

See docs/SPEC.md for the full register map and command semantics.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


# ----------------------------------------------------------------------------
# Gate-level vs. RTL build detection
# ----------------------------------------------------------------------------
# The Tiny Tapeout template sets GATES=yes when running the synthesized
# gate-level netlist (post-synth flatten). After flattening, internal
# hierarchical names like `tb.user_project.u_pe.result_r` no longer exist,
# so any test that pokes internal RTL nets via cocotb VPI handles must be
# skipped in that build. Tests that exercise the design purely through the
# SPI interface (ui_in / uio_in / uo_out / uio_out) run in both modes.
GATES = os.environ.get("GATES", "no").lower() == "yes"


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CLK_PERIOD_NS = 40        # 25 MHz nominal sysclk (1/40ns)
SCLK_HALF_CYC = 4         # sysclks per SCLK half period -> SCLK = 8 sysclks
                          # SCLK = 3.125 MHz; below f_sys/4 limit (sec 9.1)

# Register addresses (mirror src/regfile.sv)
A_STATUS      = 0x00
A_CMD         = 0x01
A_OP_A        = 0x02
A_OP_B        = 0x03
A_BIAS        = 0x04
A_QUANT_SHIFT = 0x05
A_ACT_MODE    = 0x06
A_ACC_B0      = 0x08
A_ACC_B1      = 0x09
A_ACC_B2      = 0x0A
A_ACC_B3      = 0x0B
A_RESULT      = 0x0C
A_FEATURE_ID  = 0x10
# Lane 1..3 operand registers for CMD_DOT4 (lane 0 uses A_OP_A / A_OP_B)
A_OP_A1       = 0x12
A_OP_B1       = 0x13
A_OP_A2       = 0x14
A_OP_B2       = 0x15
A_OP_A3       = 0x16
A_OP_B3       = 0x17

# CMD opcodes
CMD_NOP       = 0x00
CMD_MAC       = 0x01
CMD_CLR_ACC   = 0x02
CMD_POSTPROC  = 0x03
CMD_DOT4      = 0x04
CMD_RESET     = 0xFF

# ACT_MODE encodings (npu_types_pkg::act_mode_e)
ACT_NONE       = 0b00
ACT_RELU       = 0b01
ACT_LEAKY_RELU = 0b10

# STATUS bit positions
STATUS_IDLE         = 1 << 0
STATUS_BUSY         = 1 << 1
STATUS_RESULT_VALID = 1 << 2
STATUS_ACC_OVF_STK  = 1 << 3

FEATURE_ID_VAL = 0xA1


# ----------------------------------------------------------------------------
# SPI bit-banger helpers
# ----------------------------------------------------------------------------
def _uio(cs_n: int, sclk: int, mosi: int) -> int:
    """Pack the three SPI master-driven lines into a uio_in byte."""
    return (cs_n & 1) | ((sclk & 1) << 1) | ((mosi & 1) << 2)


def _miso(dut) -> int:
    return (int(dut.uio_out.value) >> 3) & 1


async def spi_idle(dut):
    """Park the bus in SPI mode-0 idle (CS_N high, SCLK low)."""
    dut.uio_in.value = _uio(1, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF_CYC)


async def spi_xfer_byte(dut, tx_byte: int) -> int:
    """Shift one byte MSB-first. Returns the byte received on MISO."""
    rx = 0
    for i in range(8):
        mosi_bit = (tx_byte >> (7 - i)) & 1
        # SCLK low phase: master drives MOSI; slave updates MISO on this edge
        dut.uio_in.value = _uio(0, 0, mosi_bit)
        await ClockCycles(dut.clk, SCLK_HALF_CYC)
        # SCLK high phase: slave samples MOSI on rising edge; master latches MISO
        dut.uio_in.value = _uio(0, 1, mosi_bit)
        await ClockCycles(dut.clk, SCLK_HALF_CYC)
        rx = (rx << 1) | _miso(dut)
    return rx


async def spi_write(dut, addr: int, data: int):
    """SPI write: 16-bit frame {1, addr[6:0], data[7:0]}."""
    # CS_N falling edge
    dut.uio_in.value = _uio(0, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF_CYC)
    await spi_xfer_byte(dut, 0x80 | (addr & 0x7F))
    await spi_xfer_byte(dut, data & 0xFF)
    # Trailing low half before CS_N rise
    dut.uio_in.value = _uio(0, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF_CYC)
    # CS_N rising edge -> regfile latches the write
    dut.uio_in.value = _uio(1, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF_CYC)


async def spi_read(dut, addr: int) -> int:
    """SPI read: 16-bit frame {0, addr[6:0], 8 don't-care MOSI bits}."""
    dut.uio_in.value = _uio(0, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF_CYC)
    await spi_xfer_byte(dut, 0x00 | (addr & 0x7F))
    rx = await spi_xfer_byte(dut, 0x00)
    dut.uio_in.value = _uio(0, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF_CYC)
    dut.uio_in.value = _uio(1, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF_CYC)
    return rx


async def spi_read_acc(dut) -> int:
    """Read the 32-bit accumulator. B0 first (locks the shadow), B1..3 from shadow."""
    b0 = await spi_read(dut, A_ACC_B0)
    b1 = await spi_read(dut, A_ACC_B1)
    b2 = await spi_read(dut, A_ACC_B2)
    b3 = await spi_read(dut, A_ACC_B3)
    raw = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    if raw & 0x8000_0000:
        raw -= 1 << 32
    return raw


async def wait_idle(dut, timeout: int = 64):
    """Poll STATUS until IDLE=1. Each poll itself burns >> FSM latency so this
    is just a defensive guard against accidental BUSY hangs."""
    for _ in range(timeout):
        if (await spi_read(dut, A_STATUS)) & STATUS_IDLE:
            return
    raise TimeoutError("FSM never returned to IDLE")


# ----------------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------------
async def reset_dut(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    dut.ena.value    = 1
    dut.ui_in.value  = 0
    dut.uio_in.value = _uio(1, 0, 0)  # SPI idle BEFORE deasserting reset
    dut.rst_n.value  = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value  = 1
    await ClockCycles(dut.clk, 10)


# ============================================================================
# Tests
# ============================================================================

@cocotb.test()
async def test_feature_id(dut):
    """FEATURE_ID returns 0xA1 (sec 4.1)."""
    await reset_dut(dut)
    fid = await spi_read(dut, A_FEATURE_ID)
    assert fid == FEATURE_ID_VAL, f"FEATURE_ID expected 0x{FEATURE_ID_VAL:02x}, got 0x{fid:02x}"


@cocotb.test()
async def test_status_idle_after_reset(dut):
    """STATUS = {0, 0, 0, 0, 0, 0, 0, IDLE=1} after hard reset."""
    await reset_dut(dut)
    s = await spi_read(dut, A_STATUS)
    assert s & STATUS_IDLE,               f"IDLE should be 1, status=0x{s:02x}"
    assert not (s & STATUS_BUSY),         f"BUSY should be 0, status=0x{s:02x}"
    assert not (s & STATUS_RESULT_VALID), f"RESULT_VALID should be 0, status=0x{s:02x}"
    assert not (s & STATUS_ACC_OVF_STK),  f"ACC_OVF_STK should be 0, status=0x{s:02x}"


@cocotb.test()
async def test_register_loopback(dut):
    """Configuration registers round-trip through the SPI interface."""
    await reset_dut(dut)
    cases = [
        (A_OP_A,        0x12,           0xFF),
        (A_OP_B,        0xF7,           0xFF),  # -9 signed
        (A_BIAS,        0x05,           0xFF),
        (A_QUANT_SHIFT, 0x13,           0x1F),  # only 5 LSBs are stored
        (A_ACT_MODE,    ACT_LEAKY_RELU, 0x03),
    ]
    for a, d, _ in cases:
        await spi_write(dut, a, d)
    for a, d, mask in cases:
        rd = await spi_read(dut, a)
        assert rd == (d & mask), f"reg 0x{a:02x}: expected 0x{d & mask:02x}, got 0x{rd:02x}"


@cocotb.test()
async def test_acc_starts_zero(dut):
    """Accumulator reads as zero after hard reset."""
    await reset_dut(dut)
    acc = await spi_read_acc(dut)
    assert acc == 0, f"ACC after reset expected 0, got {acc}"


@cocotb.test()
async def test_single_mac(dut):
    """CMD_MAC computes acc <- acc + OP_A * OP_B."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 10)
    await spi_write(dut, A_OP_B, 3)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    acc = await spi_read_acc(dut)
    assert acc == 30, f"ACC expected 30, got {acc}"


@cocotb.test()
async def test_accumulate_many(dut):
    """Repeated CMD_MAC accumulates correctly."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 7)
    await spi_write(dut, A_OP_B, 4)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    for _ in range(5):
        await spi_write(dut, A_CMD, CMD_MAC)
        await wait_idle(dut)
    acc = await spi_read_acc(dut)
    assert acc == 7 * 4 * 5, f"ACC expected {7 * 4 * 5}, got {acc}"


@cocotb.test()
async def test_dot4_basic(dut):
    """CMD_DOT4 runs 4 back-to-back MACs over (OP_A{0..3}, OP_B{0..3})."""
    await reset_dut(dut)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    # Lane 0 uses A_OP_A / A_OP_B; lanes 1..3 use the dedicated regs.
    pairs = [(2, 3), (5, 7), (-4, 6), (10, -8)]  # acc += 6 + 35 - 24 - 80 = -63
    await spi_write(dut, A_OP_A,  pairs[0][0] & 0xFF)
    await spi_write(dut, A_OP_B,  pairs[0][1] & 0xFF)
    await spi_write(dut, A_OP_A1, pairs[1][0] & 0xFF)
    await spi_write(dut, A_OP_B1, pairs[1][1] & 0xFF)
    await spi_write(dut, A_OP_A2, pairs[2][0] & 0xFF)
    await spi_write(dut, A_OP_B2, pairs[2][1] & 0xFF)
    await spi_write(dut, A_OP_A3, pairs[3][0] & 0xFF)
    await spi_write(dut, A_OP_B3, pairs[3][1] & 0xFF)
    await spi_write(dut, A_CMD, CMD_DOT4)
    await wait_idle(dut)
    expected = sum(a * b for a, b in pairs)
    acc = await spi_read_acc(dut)
    assert acc == expected, f"DOT4 ACC expected {expected}, got {acc}"


@cocotb.test()
async def test_dot4_lane_regs_loopback(dut):
    """Lane 1..3 operand registers round-trip through SPI."""
    await reset_dut(dut)
    cases = [
        (A_OP_A1, 0x11), (A_OP_B1, 0x22),
        (A_OP_A2, 0x33), (A_OP_B2, 0x44),
        (A_OP_A3, 0x55), (A_OP_B3, 0x66),
    ]
    for a, d in cases:
        await spi_write(dut, a, d)
    for a, d in cases:
        rd = await spi_read(dut, a)
        assert rd == d, f"reg 0x{a:02x}: expected 0x{d:02x}, got 0x{rd:02x}"


@cocotb.test()
async def test_dot4_accumulates_with_prior(dut):
    """CMD_DOT4 accumulates on top of the existing ACC, like CMD_MAC."""
    await reset_dut(dut)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    # Seed ACC with a single MAC first.
    await spi_write(dut, A_OP_A, 9)
    await spi_write(dut, A_OP_B, 4)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    seed = 9 * 4
    # Then DOT4 four MACs of 1*1.
    for a in (A_OP_A, A_OP_A1, A_OP_A2, A_OP_A3):
        await spi_write(dut, a, 1)
    for b in (A_OP_B, A_OP_B1, A_OP_B2, A_OP_B3):
        await spi_write(dut, b, 1)
    await spi_write(dut, A_CMD, CMD_DOT4)
    await wait_idle(dut)
    acc = await spi_read_acc(dut)
    assert acc == seed + 4, f"DOT4 accumulate expected {seed + 4}, got {acc}"


@cocotb.test()
async def test_signed_mac(dut):
    """OP_A and OP_B are signed INT8; product sign-extends into INT32."""
    await reset_dut(dut)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    await spi_write(dut, A_OP_A, 0xFB)  # -5
    await spi_write(dut, A_OP_B, 0x04)  # +4
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    acc = await spi_read_acc(dut)
    assert acc == -20, f"signed MAC expected -20, got {acc}"


@cocotb.test()
async def test_clr_acc(dut):
    """CMD_CLR_ACC zeroes the accumulator."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 6)
    await spi_write(dut, A_OP_B, 7)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    pre = await spi_read_acc(dut)
    assert pre == 42, f"pre-clear acc expected 42, got {pre}"
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    post = await spi_read_acc(dut)
    assert post == 0, f"post-clear acc expected 0, got {post}"


@cocotb.test()
async def test_postproc_relu_positive(dut):
    """POSTPROC: acc=10, bias=+2, shift=0, ReLU -> RESULT=12."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 5)
    await spi_write(dut, A_OP_B, 2)
    await spi_write(dut, A_BIAS, 2)
    await spi_write(dut, A_QUANT_SHIFT, 0)
    await spi_write(dut, A_ACT_MODE, ACT_RELU)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_POSTPROC)
    await wait_idle(dut)
    s = await spi_read(dut, A_STATUS)
    assert s & STATUS_RESULT_VALID, f"RESULT_VALID expected, status=0x{s:02x}"
    r = await spi_read(dut, A_RESULT)
    assert r == 12, f"RESULT expected 12, got {r}"


@cocotb.test()
async def test_postproc_relu_negative(dut):
    """POSTPROC: acc=-12, bias=0, ReLU -> RESULT=0."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 0xFD)  # -3
    await spi_write(dut, A_OP_B, 4)
    await spi_write(dut, A_BIAS, 0)
    await spi_write(dut, A_QUANT_SHIFT, 0)
    await spi_write(dut, A_ACT_MODE, ACT_RELU)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_POSTPROC)
    await wait_idle(dut)
    r = await spi_read(dut, A_RESULT)
    assert r == 0, f"ReLU(neg) expected 0, got {r}"


@cocotb.test()
async def test_postproc_with_quant_shift(dut):
    """POSTPROC: acc=100, bias=4, shift=3 -> (104 >> 3) = 13 (ACT_NONE)."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 10)
    await spi_write(dut, A_OP_B, 10)
    await spi_write(dut, A_BIAS, 4)
    await spi_write(dut, A_QUANT_SHIFT, 3)
    await spi_write(dut, A_ACT_MODE, ACT_NONE)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_POSTPROC)
    await wait_idle(dut)
    r = await spi_read(dut, A_RESULT)
    assert r == 13, f"quantised result expected 13, got {r}"


@cocotb.test()
async def test_result_valid_clears_on_read(dut):
    """Reading RESULT clears RESULT_VALID (sec 5.4)."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 1)
    await spi_write(dut, A_OP_B, 1)
    await spi_write(dut, A_BIAS, 0)
    await spi_write(dut, A_QUANT_SHIFT, 0)
    await spi_write(dut, A_ACT_MODE, ACT_NONE)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_POSTPROC)
    await wait_idle(dut)
    s1 = await spi_read(dut, A_STATUS)
    assert s1 & STATUS_RESULT_VALID, f"RESULT_VALID should be set, status=0x{s1:02x}"
    _ = await spi_read(dut, A_RESULT)
    s2 = await spi_read(dut, A_STATUS)
    assert not (s2 & STATUS_RESULT_VALID), \
        f"RESULT_VALID should clear after RESULT read, status=0x{s2:02x}"


@cocotb.test()
async def test_cmd_reset_preserves_config(dut):
    """CMD_RESET clears acc/RESULT/sticky flags but preserves config (sec 6.1, 8.2)."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A,        0x55)
    await spi_write(dut, A_OP_B,        0xAA)
    await spi_write(dut, A_BIAS,        0x11)
    await spi_write(dut, A_QUANT_SHIFT, 0x1A)
    await spi_write(dut, A_ACT_MODE,    ACT_LEAKY_RELU)
    # Dirty the accumulator
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    assert (await spi_read_acc(dut)) != 0, "MAC should have changed acc"
    # Soft reset
    await spi_write(dut, A_CMD, CMD_RESET)
    await wait_idle(dut)
    # Config preserved
    assert (await spi_read(dut, A_OP_A))        == 0x55
    assert (await spi_read(dut, A_OP_B))        == 0xAA
    assert (await spi_read(dut, A_BIAS))        == 0x11
    assert (await spi_read(dut, A_QUANT_SHIFT)) == 0x1A
    assert (await spi_read(dut, A_ACT_MODE))    == ACT_LEAKY_RELU
    # Accumulator and result cleared
    assert (await spi_read_acc(dut)) == 0, "ACC should be 0 after CMD_RESET"
    s = await spi_read(dut, A_STATUS)
    assert not (s & STATUS_RESULT_VALID), "RESULT_VALID should be 0 after CMD_RESET"
    assert not (s & STATUS_ACC_OVF_STK),  "ACC_OVF_STK should be 0 after CMD_RESET"


@cocotb.test()
async def test_hard_reset_clears_all(dut):
    """rst_n pulse clears every register including the config registers (sec 8.2)."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 0x7E)
    await spi_write(dut, A_OP_B, 0x11)
    await spi_write(dut, A_BIAS, 0x22)
    # Pulse hard reset
    dut.uio_in.value = _uio(1, 0, 0)
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 10)
    assert (await spi_read(dut, A_OP_A)) == 0
    assert (await spi_read(dut, A_OP_B)) == 0
    assert (await spi_read(dut, A_BIAS)) == 0


@cocotb.test()
async def test_acc_spans_multiple_bytes(dut):
    """Accumulator reads coherently across all four bytes via the shadow."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 100)
    await spi_write(dut, A_OP_B, 100)
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    # 100*100 = 10_000 per MAC; 200 MACs -> 2_000_000 (0x001E_8480) spans 3 bytes
    for _ in range(200):
        await spi_write(dut, A_CMD, CMD_MAC)
        await wait_idle(dut)
    acc = await spi_read_acc(dut)
    assert acc == 100 * 100 * 200, f"ACC expected {100*100*200}, got {acc}"


@cocotb.test()
async def test_nop_and_unknown_cmd(dut):
    """CMD_NOP and unknown opcodes leave the accumulator untouched."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 3)
    await spi_write(dut, A_OP_B, 5)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    acc_before = await spi_read_acc(dut)
    await spi_write(dut, A_CMD, CMD_NOP)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, 0x7E)  # unknown opcode
    await wait_idle(dut)
    acc_after = await spi_read_acc(dut)
    assert acc_after == acc_before, \
        f"NOP/unknown should not change acc; before={acc_before}, after={acc_after}"


@cocotb.test()
async def test_miso_idle_low(dut):
    """MISO is driven to 0 when CS_N is high; uio_oe[3] = 1 (sec 3.1, 7.3)."""
    await reset_dut(dut)
    dut.uio_in.value = _uio(1, 0, 0)
    await ClockCycles(dut.clk, 8)
    assert _miso(dut) == 0, f"MISO should be 0 when CS_N high, got {_miso(dut)}"
    assert (int(dut.uio_oe.value) >> 3) & 1, \
        f"uio_oe[3] should be 1, got uio_oe=0x{int(dut.uio_oe.value):02x}"


# ----------------------------------------------------------------------------
# Sticky-overflow and quantizer-saturation coverage (T9, T13, T14).
#
# Reaching the INT32 overflow boundary or the INT8 saturation boundary by
# accumulating real MACs would take >2^17 SPI transactions, which is not
# feasible in simulation. We instead deposit a chosen accumulator value
# directly into conv_pe.result_r via cocotb's VPI handle, then issue a
# single CMD_MAC (or CMD_POSTPROC) to exercise the path under test.
# ----------------------------------------------------------------------------


def _force_pe_acc(dut, value: int):
    """Deposit a 32-bit value into the PE accumulator register.

    Reaches into the RTL hierarchy via cocotb VPI. This only works against
    the unsynthesized RTL build; after gate-level synthesis the design is
    flattened and `u_pe.result_r` no longer exists as a named handle.
    Tests that call this helper must be guarded with `skip=GATES`.
    """
    dut.user_project.u_pe.result_r.value = value & 0xFFFF_FFFF


@cocotb.test(skip=GATES)
async def test_acc_overflow_sticks(dut):
    """T9: a MAC whose signed sum overflows INT32 sets ACC_OVF_STK, and the
    flag stays set across subsequent non-overflowing MACs."""
    await reset_dut(dut)
    await spi_write(dut, A_OP_A, 1)
    await spi_write(dut, A_OP_B, 1)
    # Park acc one short of INT32_MAX so the +1 MAC overflows.
    _force_pe_acc(dut, 0x7FFF_FFFF)
    await ClockCycles(dut.clk, 1)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    s = await spi_read(dut, A_STATUS)
    assert s & STATUS_ACC_OVF_STK, \
        f"ACC_OVF_STK should set after INT32 overflow, status=0x{s:02x}"
    # Sticky: a clean MAC/CLR cycle must not clear it.
    await spi_write(dut, A_CMD, CMD_CLR_ACC)
    await wait_idle(dut)
    await spi_write(dut, A_CMD, CMD_MAC)
    await wait_idle(dut)
    s2 = await spi_read(dut, A_STATUS)
    assert s2 & STATUS_ACC_OVF_STK, \
        f"ACC_OVF_STK should remain sticky, status=0x{s2:02x}"
    # Hard reset clears the sticky flag.
    dut.uio_in.value = _uio(1, 0, 0)
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 10)
    s3 = await spi_read(dut, A_STATUS)
    assert not (s3 & STATUS_ACC_OVF_STK), \
        f"ACC_OVF_STK should clear on hard reset, status=0x{s3:02x}"


@cocotb.test(skip=GATES)
async def test_quantize_saturate_high(dut):
    """T13: a large positive ACC clamps to +127 at the quantizer output."""
    await reset_dut(dut)
    await spi_write(dut, A_BIAS, 0)
    await spi_write(dut, A_QUANT_SHIFT, 0)
    await spi_write(dut, A_ACT_MODE, ACT_NONE)
    # 16384 >> 0 = 16384, far above INT8_MAX -> clamps to +127 (0x7F).
    _force_pe_acc(dut, 0x0000_4000)
    await ClockCycles(dut.clk, 1)
    await spi_write(dut, A_CMD, CMD_POSTPROC)
    await wait_idle(dut)
    r = await spi_read(dut, A_RESULT)
    assert r == 0x7F, f"saturate-high expected 0x7F, got 0x{r:02x}"


@cocotb.test(skip=GATES)
async def test_quantize_saturate_low(dut):
    """T14: a large negative ACC clamps to -128 at the quantizer output."""
    await reset_dut(dut)
    await spi_write(dut, A_BIAS, 0)
    await spi_write(dut, A_QUANT_SHIFT, 0)
    await spi_write(dut, A_ACT_MODE, ACT_NONE)
    # -16384 >> 0 = -16384, far below INT8_MIN -> clamps to -128 (0x80).
    _force_pe_acc(dut, (-16384) & 0xFFFF_FFFF)
    await ClockCycles(dut.clk, 1)
    await spi_write(dut, A_CMD, CMD_POSTPROC)
    await wait_idle(dut)
    r = await spi_read(dut, A_RESULT)
    assert r == 0x80, f"saturate-low expected 0x80, got 0x{r:02x}"

