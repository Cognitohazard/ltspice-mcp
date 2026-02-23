"""MCP prompt handlers for guided SPICE circuit design workflows."""

import logging

from mcp import types

logger = logging.getLogger(__name__)


def get_prompt_definitions() -> list[types.Prompt]:
    """Return all available prompt definitions."""
    return [
        types.Prompt(
            name="filter_design",
            description="Guided workflow for designing and simulating analog filters in LTspice",
            arguments=[
                types.PromptArgument(name="filter_type", description="Filter type: lowpass, highpass, bandpass, bandstop, or allpass", required=True),
                types.PromptArgument(name="target_frequency", description="Target cutoff or center frequency (e.g. '1kHz', '100Hz', '10MHz')", required=True),
                types.PromptArgument(name="order", description="Filter order (e.g. 1, 2, 4). Higher order = steeper rolloff", required=False),
                types.PromptArgument(name="topology", description="Filter topology: RC, RLC, Sallen-Key, MFB, state-variable, or cascaded-biquad", required=False),
                types.PromptArgument(name="specs", description="Additional specs: passband ripple, stopband attenuation, source/load impedance, supply voltage", required=False),
            ],
        ),
        types.Prompt(
            name="amplifier_analysis",
            description="Guided workflow for analyzing amplifier bias, gain, bandwidth, and stability in LTspice",
            arguments=[
                types.PromptArgument(name="topology", description="Amplifier topology: common-emitter, common-source, inverting-opamp, non-inverting-opamp, differential, etc.", required=True),
                types.PromptArgument(name="supply_voltage", description="Supply voltage(s) (e.g. '12V', '+/-15V', '3.3V')", required=True),
                types.PromptArgument(name="specs", description="Target specs: gain (dB or V/V), bandwidth, output swing, input impedance", required=False),
                types.PromptArgument(name="components", description="Key component values or transistor/op-amp model if already chosen", required=False),
            ],
        ),
        types.Prompt(
            name="tolerance_analysis",
            description="Guided workflow for Monte Carlo tolerance analysis and yield estimation in LTspice",
            arguments=[
                types.PromptArgument(name="circuit", description="Circuit netlist filename or description", required=True),
                types.PromptArgument(name="parameters_of_interest", description="Output parameters to measure: gain, cutoff frequency, output voltage, etc.", required=True),
                types.PromptArgument(name="num_runs", description="Number of Monte Carlo runs (default 500; 100 for screening, 1000+ for high confidence)", required=False),
                types.PromptArgument(name="tolerance_specs", description="Component tolerance specs (e.g. 'resistors 1%, capacitors 10%') and acceptable output variation", required=False),
            ],
        ),
        types.Prompt(
            name="simulation_debugging",
            description="Guided workflow for diagnosing and fixing LTspice simulation errors and unexpected results",
            arguments=[
                types.PromptArgument(name="problem_description", description="Description of the problem or error message", required=True),
                types.PromptArgument(name="circuit", description="Circuit netlist filename or description", required=False),
                types.PromptArgument(name="symptoms", description="Observable symptoms: error messages, wrong values, simulation hangs, convergence failures", required=False),
            ],
        ),
    ]


async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
    """Dispatch prompt request by name to the appropriate builder."""
    args = arguments or {}

    if name == "filter_design":
        return _build_filter_design_prompt(args)
    elif name == "amplifier_analysis":
        return _build_amplifier_analysis_prompt(args)
    elif name == "tolerance_analysis":
        return _build_tolerance_analysis_prompt(args)
    elif name == "simulation_debugging":
        return _build_simulation_debugging_prompt(args)
    else:
        raise ValueError(f"Unknown prompt name: '{name}'. Available: filter_design, amplifier_analysis, tolerance_analysis, simulation_debugging")


# ---------------------------------------------------------------------------
# Private builder functions
# ---------------------------------------------------------------------------


def _build_filter_design_prompt(args: dict[str, str]) -> types.GetPromptResult:
    filter_type = args.get("filter_type", "not specified")
    target_frequency = args.get("target_frequency", "not specified")
    order = args.get("order", "not specified")
    topology = args.get("topology", "not specified")
    specs = args.get("specs", "none provided")

    prompt_text = f"""# Filter Design Workflow

## Your Design Task

- Filter type: {filter_type}
- Target frequency: {target_frequency}
- Filter order: {order}
- Topology: {topology}
- Additional specs: {specs}

## Before You Begin — Clarifying Questions

If any of the following are not specified, **ask the user before proceeding**:

1. What filter order is required? (If not given, ask: "Do you need a 1st, 2nd, or higher-order filter? Higher order gives steeper rolloff but more components.")
2. What are the source and load impedances? (Affects component scaling and whether a buffer is needed.)
3. For active topologies: What supply voltage is available? Single or dual supply?
4. Is passband flatness (Butterworth), equal ripple (Chebyshev), or linear phase (Bessel) required?
5. What is the required stopband attenuation, if any?

Do not assume defaults silently. Surface trade-offs so the user can decide.

---

## Step 1: Topology Selection

Choose the right topology based on the requirements:

**Passive topologies:**

- **RC (1st order):** -20 dB/decade rolloff. fc = 1/(2πRC). Simple but limited to 1st order without buffering. No supply needed.
- **RLC (2nd order):** fc = 1/(2π√LC), Q = (1/R)√(L/C). Can achieve any Q. Use passive for RF/high-frequency (>100 MHz). Inductors are large and lossy at audio frequencies.

**Active topologies (require op-amp and supply):**

- **Sallen-Key:** Non-inverting (unity or low gain), good for Q < 10. Butterworth 2nd order: use equal component values, K=1 (unity gain). Component spread is low. Sensitive to op-amp GBW at high Q. Use for fc < 1 MHz.
- **Multiple Feedback (MFB):** Inverting, lower sensitivity to component tolerances than Sallen-Key. Preferred for bandpass with Q < 20. GBW requirement: op-amp GBW > 100× fc for Q < 5.
- **State Variable / KHN:** Best for Q > 10. Simultaneously outputs lowpass, highpass, and bandpass. Uses 3 op-amps. Most robust to component variations.
- **Cascaded biquads:** For orders > 2. Cascade 2nd-order sections. Pair poles with lowest Q first for best dynamic range.

**Frequency guidelines:**
- fc < 1 MHz: Any active topology works
- 1–100 MHz: MFB or passive LC preferred; op-amp GBW must be > 50× fc
- fc > 100 MHz: Passive LC only (op-amp limitations)

---

## Step 2: Component Value Calculation

Use `create_netlist` to start the circuit, then `set_component_value` for each component.

**Resistor selection:** Prefer E24 (5%) or E96 (1%) series values. For signal-path resistors, 1 kΩ–100 kΩ range balances noise and capacitor size. Below 1 kΩ loads the op-amp output; above 100 kΩ increases noise.

**Capacitor selection:**
- NP0/C0G: Best stability, use for timing/filter critical components, limited to ~1 nF at reasonable size
- X7R: Good for 1 nF–10 µF, ±10–15% tolerance
- Avoid Y5V/Z5U in filter circuits (large capacitance variation with voltage and temperature)

**Op-amp selection:** GBW (gain-bandwidth product) must be > 50× target cutoff frequency. For unity-gain Sallen-Key, GBW > 50× fc. For MFB with gain = 10, GBW > 500× fc.

**Sallen-Key 2nd-order Butterworth lowpass (equal component values):**
- R1 = R2 = R, C1 = C2 = C
- fc = 1/(2πRC), Gain = 1 (unity). Set Q = 0.707 with K=1.

**MFB 2nd-order lowpass:**
- fc = 1/(2π) × √(1/(R1×R3×C1×C2))
- Q = (π×fc) × (C1×C2×R1×R3)^(1/2) × (R1+R3)^(-1)

---

## Step 3: Simulation Setup

**AC frequency sweep (primary verification):**

Add to netlist: `.ac dec 100 {{target_frequency_decade_below}} {{target_frequency_decade_above}}`
Example for 1 kHz filter: `.ac dec 100 10 100k`

Use `run_simulation` to execute. The `.ac` analysis sweeps frequency and gives magnitude/phase response.

**Transient verification (optional but recommended):**

Add: `.tran 0 {{10_periods}} 0 {{period/1000}}`
Apply a sine wave at the cutoff frequency and verify -3 dB attenuation. Apply a step and observe the transient response (overshoot indicates Q > 0.707).

---

## Step 4: Result Verification

Use `list_signals` to find the output node name, then `get_signal_data` to retrieve the AC response.

**Check each of the following:**

1. **-3 dB frequency:** Use `query_value` at the target frequency. Output should be -3 dB (0.707× input) at fc. A significant deviation means component values need adjustment.

2. **Passband flatness:** For frequencies well below fc (lowpass), gain should be flat within ±0.1 dB for Butterworth, or within the specified ripple for Chebyshev.

3. **Rolloff rate:** At 10× fc (one decade above cutoff for lowpass):
   - 1st order: -20 dB/decade → -20 dB at 10× fc
   - 2nd order: -40 dB/decade → -40 dB at 10× fc
   - 4th order: -80 dB/decade → -80 dB at 10× fc
   General: -20N dB/decade for an Nth-order filter.

4. **Phase response:** At fc, phase shift should be -45° (1st order), -90° (2nd order), -180° (4th order).

5. **Q factor verification (for 2nd order):** Measure the peak in the response (if any) near fc. Q > 0.707 produces peaking. For Butterworth, there should be no peaking; Q = 0.707.

---

## Step 5: Iteration and Refinement

- **Frequency too high/low:** Scale all R values (or all C values) proportionally. If fc is off by factor k, multiply all R by 1/k.
- **Rolloff not steep enough:** Increase filter order. Each additional order adds -20 dB/decade.
- **Too much ripple/peaking:** Reduce Q (for Sallen-Key, increase feedback resistor ratio or use equal components for Butterworth Q=0.707).
- **Passband not flat enough:** Use Butterworth approximation (maximally flat, no ripple).
- **Linear phase required (audio/data applications):** Use Bessel filter coefficients; accept slower rolloff in exchange for flat group delay.
"""

    return types.GetPromptResult(
        description=f"Filter design workflow for {filter_type} filter at {target_frequency}",
        messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=prompt_text))],
    )


def _build_amplifier_analysis_prompt(args: dict[str, str]) -> types.GetPromptResult:
    topology = args.get("topology", "not specified")
    supply_voltage = args.get("supply_voltage", "not specified")
    specs = args.get("specs", "none provided")
    components = args.get("components", "none provided")

    prompt_text = f"""# Amplifier Analysis Workflow

## Your Analysis Task

- Topology: {topology}
- Supply voltage: {supply_voltage}
- Target specs: {specs}
- Components/models: {components}

## Before You Begin — Clarifying Questions

If the circuit netlist is not yet created or any of the following are missing, **ask the user before proceeding**:

1. Is the circuit netlist already created, or do you need to create it from scratch?
2. What transistor or op-amp model should be used? (Model accuracy is critical for bias and frequency response.)
3. What is the intended signal frequency range?
4. What load resistance is connected to the output?
5. For BJT/MOSFET designs: What quiescent current (Iq) is targeted?

Do not guess transistor models or bias points. Ask.

---

## Step 1: DC Bias Point Analysis (.op)

**Purpose:** Verify the amplifier is biased correctly in the active/linear region before any AC analysis.

Run: `get_operating_point` on the circuit.

**BJT bias verification:**
- VCE > 0.3 V: transistor is in active (linear) region. VCE < 0.3 V = saturation (clipping, no amplification).
- VBE ≈ 0.6–0.7 V for silicon BJT. Values outside this range indicate bias network problem.
- IC = β × IB. If β is uncertain, verify the circuit works over β range 50–300 (use `.step param` sweep).
- Common-emitter: VC should be roughly mid-rail for maximum output swing. Target VCE ≈ VCC/2.

**MOSFET bias verification:**
- VGS > Vth: transistor is ON. Vth depends on device; check the model (typically 0.5–2 V for enhancement-mode NMOS).
- VDS > VGS - Vth: transistor is in saturation (active region for amplification). VDS < VGS - Vth = triode region (resistive, not amplifying).
- ID is set by VGS via the square-law: ID ≈ (k/2)(VGS - Vth)^2 in saturation.

**Op-amp bias verification:**
- Verify input common-mode voltage is within the specified CM input range for the op-amp model.
- Output voltage should not be clipping (check against supply rails; rail-to-rail op-amps can approach but not reach the rails under load).
- Both inputs should be at nearly the same voltage in a feedback configuration (virtual short).

---

## Step 2: AC Small-Signal Analysis (.ac)

**Purpose:** Measure voltage gain and bandwidth.

Add to netlist: `.ac dec 100 1 {{appropriate_upper_frequency}}`
Use `run_simulation`. Probe the output node.

**Voltage gain formulas (for manual verification):**

- BJT common-emitter (CE): Av = -gm × (RC || RL), where gm = IC / VT (VT = 26 mV at room temp)
- MOSFET common-source (CS): Av = -gm × (RD || RL), where gm = 2ID / (VGS - Vth)
- Op-amp inverting: Av = -Rf / Rin (closed-loop, frequency-independent up to GBW/|Av|)
- Op-amp non-inverting: Av = 1 + Rf / Rg (closed-loop)
- Differential pair: Av = gm × (RC || RL) per side (differential output)

**Check in simulation:**
1. Midband gain (flat region): should match formula within ~10% for ideal models.
2. Lower -3 dB frequency (fL): set by coupling and bypass capacitor RC time constants.
3. Upper -3 dB frequency (fH): set by transistor fT, Miller capacitance, or op-amp GBW.
4. Bandwidth = fH - fL (for audio: target 20 Hz – 20 kHz minimum).

**Phase margin (critical for stability):**
- Plot phase of the loop gain (or open-loop gain for op-amp).
- Phase margin > 45° is required for stability; > 60° is preferred for acceptable transient overshoot.
- Phase margin < 30° will show significant peaking; < 0° = oscillation.

---

## Step 3: Transient Analysis

**Purpose:** Verify time-domain behavior and check for stability issues.

Add: `.tran 0 {{10_signal_periods}} 0 {{period/100}}`
Apply a sine wave at midband frequency at the expected input amplitude.

**Check:**
- Rise time ≈ 0.35 / fH (bandwidth-limited step response). Much slower = gain-bandwidth problem.
- Slew rate limit: for op-amps, if the output cannot slew fast enough, the output is distorted at high frequencies. Slew rate = maximum dVout/dt. Sine wave requires SR > 2π × f × Vpeak.
- Overshoot / ringing on a step response: overshoot > 30% or sustained ringing indicates poor phase margin (< 45°). Address by adding compensation.

---

## Step 4: Stability Analysis

**Why stability matters:** Feedback amplifiers can oscillate if loop gain phase shift reaches 180° while loop gain magnitude is still ≥ 1 (0 dB).

**Checking stability:**

1. **Phase margin method:** Open the feedback loop (break at one node), inject an AC signal, and measure the loop gain T(jω). Phase margin = 180° + phase(T) at the frequency where |T| = 1 (0 dB).

2. **Gain margin:** Amount of additional gain (in dB) before oscillation at the phase crossover frequency (where phase = -180°). Gain margin > 6 dB is required; > 10 dB is preferred.

**Compensation techniques for poor stability:**

- **Series input resistor:** Add R (10–100 Ω) between op-amp output and capacitive load. Adds a zero that improves phase margin.
- **Feedback capacitor (lead compensation):** Place a small capacitor (10–100 pF) across the feedback resistor Rf. Creates a zero in the feedback path, boosting phase margin.
- **Dominant pole compensation:** Reduce bandwidth intentionally by adding a large capacitor at a high-impedance node. Standard in internally compensated op-amps.

---

## Step 5: Robustness Sweeps

Verify performance over operating range using LTspice `.step` parameter sweeps.

**Temperature sweep:**
Add: `.step temp -40 85 25`
Check that bias point (VCE or VDS), gain, and bandwidth remain within spec across -40°C to +85°C.

**Supply voltage sweep:**
Vary VCC/VDD by ±10% and verify gain and output swing do not degrade unacceptably.

**BJT beta (β) sweep:**
Add: `.step param beta 50 300 50` and reference `{{beta}}` in transistor model.
Verify bias point is stable across the beta range. If it shifts significantly, add emitter degeneration.

**MOSFET threshold voltage sweep:**
Vth varies with process. Sweep VGS by ±0.2 V and verify ID and small-signal gm remain acceptable.
"""

    return types.GetPromptResult(
        description=f"Amplifier analysis workflow for {topology} with {supply_voltage} supply",
        messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=prompt_text))],
    )


def _build_tolerance_analysis_prompt(args: dict[str, str]) -> types.GetPromptResult:
    circuit = args.get("circuit", "not specified")
    parameters_of_interest = args.get("parameters_of_interest", "not specified")
    num_runs = args.get("num_runs", "500")
    tolerance_specs = args.get("tolerance_specs", "none provided")

    prompt_text = f"""# Tolerance Analysis (Monte Carlo) Workflow

## Your Analysis Task

- Circuit: {circuit}
- Parameters of interest: {parameters_of_interest}
- Number of runs: {num_runs}
- Tolerance specs: {tolerance_specs}

## Before You Begin — Clarifying Questions

If any of the following are missing, **ask the user before proceeding**:

1. What component tolerances apply? (Standard defaults: resistors ±1% or ±5%, capacitors ±5% or ±10% or ±20%. Do not assume without asking.)
2. What statistical distribution should be used? Uniform (worst-case conservative) or Gaussian/normal (realistic manufacturing)?
3. What is the acceptable variation in the output parameters? (e.g., "cutoff frequency within ±5%", "gain within ±0.5 dB")
4. Is there a yield target? (Consumer: >95%, Industrial: >99%, Mil/Aero: >99.9%)
5. Which components should be varied? All passives? Or specific critical components only?

Do not assume tolerances or yield targets. These define the entire analysis.

---

## Understanding Distributions

**Uniform distribution:**
- Every value within [nominal - tolerance, nominal + tolerance] is equally likely.
- More conservative (pessimistic) — gives worst-case-like bounds.
- Use for: safety-critical circuits, first-pass screening, when actual process distribution is unknown.
- If all components are at their extreme values simultaneously, that is the absolute worst case (corner analysis, not Monte Carlo).

**Gaussian (normal) distribution:**
- Component values cluster around nominal; values near the extreme tolerance are very rare.
- Standard assumption: tolerance limit = ±3σ (so 99.73% of parts are within tolerance).
- More realistic for commercial manufacturing processes.
- Use for: yield estimation, commercial products, when process Cpk data is available.
- A 3σ Gaussian analysis gives true statistical yield (probability of meeting spec).

**Recommendation:** Run both. Uniform gives a conservative bound; Gaussian gives a realistic yield estimate.

---

## Step 1: Nominal Circuit Verification

Before any Monte Carlo run, **always verify the nominal circuit works correctly**:

1. Use `run_simulation` on the circuit at nominal component values.
2. Use `get_simulation_summary` to confirm no errors or warnings.
3. Measure the nominal output parameters using `get_signal_data` or `query_value`.
4. Record the nominal values — all Monte Carlo results are interpreted relative to these.

**If the nominal simulation fails, do not proceed to Monte Carlo.** Fix the circuit first (see simulation_debugging prompt).

**Add measurement directives** if not already present. Use `add_instruction` to add `.MEAS` statements:
- Example: `.MEAS AC fc WHEN V(out)=0.707*V(out,0)` (measures -3 dB frequency)
- Example: `.MEAS TRAN vout_peak MAX V(out)`

---

## Step 2: Configure Monte Carlo

Use the `configure_montecarlo` tool to set up the analysis.

**Key parameters:**

- `distribution`: "uniform" or "gaussian"
- `num_runs`:
  - 100 runs: Screening only. Mean and rough bounds. Yield estimate ±5-10%.
  - 500 runs: Good statistical confidence. Yield estimate ±2-3%. Recommended starting point.
  - 1000+ runs: High confidence yield estimates. Required for >99% yield claims.
  - For 99.9% yield (mil/aero), use ≥ 5000 runs.
- `seed`: Set an integer seed for reproducibility (e.g., `seed=42`). Allows re-running identical analysis.
- `type_tolerances`: Dictionary of component type to tolerance percentage (e.g., `{{"R": 1, "C": 5}}`)

**Rule of thumb:** If you need to claim X% yield, you need at least 10/(1-X/100) runs. For 99% yield claim: 1000 runs minimum.

---

## Step 3: Analyze Results

Use `check_batch_job` to monitor progress, then `get_batch_results` when complete.

**Interpret the statistical output:**

1. **Mean (µ):** Average output across all runs. Compare to nominal. A large offset from nominal suggests a systematic bias (e.g., loading effect that nominal simulation doesn't capture).

2. **Standard deviation (σ):** Spread of results.
   - 99.7% of outputs fall within µ ± 3σ (for Gaussian inputs and linear circuits).
   - 1σ = 68.3%, 2σ = 95.4%, 3σ = 99.7%.

3. **Yield:** Fraction of runs where the output parameter is within specification.
   - `yield = (runs within spec) / (total runs) × 100%`
   - Compare against your yield target.

4. **Min / Max:** Absolute worst and best case across all Monte Carlo runs. These are not the true worst case (that requires corner analysis) but give practical bounds.

**Sensitivity analysis:** Which component affects the output the most? Use `get_batch_results` with parameter correlation to identify the highest-sensitivity components. The component whose variation explains the most output variance is your yield-limiting component.

---

## Step 4: Identify Worst-Case Contributors

Use `get_batch_results` filtered to the failing runs (runs where output is outside spec).

**For each failing run:**
- What component values deviated most from nominal?
- Is there a pattern? (e.g., all failures have R1 at its high extreme and C2 at its low extreme)

**Correlation analysis:**
- If output_parameter vs. component_value scatter plot shows a tight correlation, that component dominates the yield loss.
- Rank components by |∂output/∂component| × tolerance. Largest = most critical.

**Corner analysis (complementary to Monte Carlo):**
After identifying sensitive components, run corner simulations at ±tolerance extremes for just those components. This confirms worst-case behavior deterministically.

---

## Step 5: Improve Yield

**Tighten sensitive component tolerance:**
- If R1 is the dominant contributor, switching from 5% to 1% tolerances often raises yield from 90% to >99%.
- Cost trade-off: 1% resistors cost ~2–5× more than 5%. Evaluate whether yield gain justifies cost.

**Add negative feedback:**
- Feedback desensitizes the output to component variations.
- Example: emitter degeneration in a BJT amplifier reduces gain sensitivity to β.
- Op-amp closed-loop gain is set by resistor ratios, not device parameters.

**Investigate bimodal distributions:**
- If the output histogram shows two peaks, there is a threshold effect — some runs operate in a fundamentally different regime (e.g., oscillator vs. non-oscillator, comparator flipping).
- This is a design problem, not a tolerance problem. The nominal design is near a bifurcation point and must be redesigned.

---

## Common Pitfalls

1. **Too few runs for yield claims:** Claiming 99% yield from 100 runs is statistically meaningless. Match run count to yield requirement.
2. **Ignoring temperature variation:** Component values change with temperature. Combine Monte Carlo with temperature sweep for realistic production yield.
3. **Not checking nominal first:** Monte Carlo on a broken circuit wastes time. Always verify nominal simulation passes first.
4. **Using uniform distribution for yield claims:** Uniform over-estimates worst case. Use Gaussian for realistic yield; use uniform for margin verification.
5. **Varying all components equally:** Not all components contribute equally. Identify critical components and focus tolerance budget there.
"""

    return types.GetPromptResult(
        description=f"Monte Carlo tolerance analysis for {circuit}, measuring {parameters_of_interest}",
        messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=prompt_text))],
    )


def _build_simulation_debugging_prompt(args: dict[str, str]) -> types.GetPromptResult:
    problem_description = args.get("problem_description", "not specified")
    circuit = args.get("circuit", "not specified")
    symptoms = args.get("symptoms", "not specified")

    prompt_text = f"""# Simulation Debugging Workflow

## Problem Summary

- Problem: {problem_description}
- Circuit: {circuit}
- Symptoms: {symptoms}

## Before You Begin — Clarifying Questions

If any of the following are missing, **ask the user before proceeding**:

1. What is the **exact error message** from the LTspice log? (Copy the full text, not a paraphrase.)
2. What netlist filename is being simulated?
3. What simulation type is being used? (.op, .ac, .tran, .dc)
4. What is the expected result vs. what is actually observed?

Debugging without the exact error message is guesswork. Always ask.

---

## Category 1: Convergence Failures

**Symptoms:** "timestep too small", "singular matrix", "no convergence in DC operating point", "internal time step too small"

These errors mean the simulator's numerical solver could not find a solution. Work through this checklist in order:

### 1. Check for a ground node (Node 0)
Every circuit MUST have at least one connection to node 0 (ground). Without it, the node voltage matrix is singular and cannot be solved.
- Use `read_netlist` to verify a ground connection exists.
- Every sub-circuit must have a DC path to ground (even if AC-coupled, a bias resistor to ground is often needed).

### 2. Check for floating nodes
A floating node has no DC path to ground (connected only through capacitors or open circuits). This creates an undefined DC voltage.
- **MOSFET/BJT gates/bases with only capacitive coupling:** Add a large bias resistor (1 GΩ is typical: `Rbias out 0 1G`) to set the DC operating point without affecting AC response.
- **Capacitor-only connections:** Add a 1 GΩ resistor from the floating node to ground.
- Use `read_netlist` and trace every node to verify it has a DC path to ground.

### 3. Check for voltage source loops
Two ideal voltage sources connected in a loop (or a voltage source directly across a short) create an infinite current and a singular matrix.
- Add a small series resistance (1 mΩ = 0.001 Ω) in series with each voltage source: `R_series Vplus Vnode 0.001`
- Also check for inductors in parallel with voltage sources (same issue at DC).

### 4. Adjust solver options
Add these to the netlist via `add_instruction` one at a time, re-running after each:

```
.options reltol=0.01          ; Relax relative tolerance (default 0.001, try 0.01)
.options abstol=1e-10         ; Absolute current tolerance
.options vntol=1e-4           ; Absolute voltage tolerance
.options gmin=1e-10           ; Minimum conductance (prevents singular matrix)
.options method=gear          ; Use Gear integration method (better for stiff circuits)
.options cshunt=1e-15         ; Add tiny capacitor to every node (improves convergence)
```

Try `reltol=0.01` first — this resolves most convergence issues in circuits with very different impedance scales.

### 5. Set initial conditions
For circuits with multiple stable operating points (flip-flops, oscillators, bistable circuits), the solver may not converge to the desired state.
- Add `.ic V(node)=voltage` to set initial node voltages.
- Example: `.ic V(out)=0 V(vdd)=5` to start from a known state.

### 6. Replace step sources with ramps
Ideal step voltage/current sources (0 → V in 0 time) inject infinite current at t=0 and can cause convergence failures.
- Use a ramp: `PULSE(0 5 0 1n 1n 1m 10m)` (1 ns rise time instead of 0).
- Apply to any source with a zero transition time.

---

## Category 2: Wrong Results

**Symptoms:** Gain is wrong, frequency response is shifted, DC output is unexpected, signal is missing

### A. Verify circuit connections
Use `read_netlist` to check:
- Are all component nodes connected to the intended nets?
- Are there any typos in node names? (LTspice node names are case-insensitive but a space creates a new node)

### B. Check SPICE notation traps (very common source of errors)
LTspice uses non-standard unit multipliers:

| You write | SPICE interprets |
|-|-|
| 1M | 1 milli (0.001) — NOT 1 Mega |
| 1Meg | 1 Mega (1,000,000) — correct |
| 1m | 1 milli (0.001) |
| 1u | 1 micro (0.000001) |
| 1n | 1 nano |
| 1p | 1 pico |

**Never use "1M" to mean 1 MΩ or 1 MHz — it means 1 mΩ or 1 mHz in SPICE.**

Also: no space between value and unit. Write `1k` not `1 k`.

### C. Verify DC bias before AC analysis
Use `get_operating_point` to check DC bias. A wrongly biased circuit will give wrong AC results.

### D. Check AC source syntax
For `.ac` analysis, the AC voltage source must include the `AC` keyword:
- Correct: `V1 in 0 AC 1` (sets AC amplitude to 1 V for small-signal analysis)
- Wrong: `V1 in 0 1` (this is a DC source, not an AC source — output will be 0 in .ac)

### E. Verify device models
Use `get_model_info` to confirm the transistor or op-amp model is loaded correctly. An incorrect or missing model causes the simulator to use default parameters (often wrong).

---

## Category 3: Simulation Hangs

**Symptoms:** Simulation runs indefinitely without completing

### A. Oscillating circuit
A circuit that oscillates will run forever in transient simulation (the solver keeps time-stepping).
- Add `.options maxstep={{period/20}}` where period is the expected oscillation period.
- If the circuit should NOT oscillate, check for stability (see amplifier_analysis prompt).

### B. Transient simulation end time too long
Check the `.tran` statement. If the stop time is `1s` but the circuit operates at 1 MHz (period = 1 µs), the simulation runs 1,000,000 cycles — which takes a very long time.
- Reduce the stop time to 10–20 cycles of the lowest frequency of interest.

### C. Extreme component values
Very large or very small component values create extreme time constants that force the simulator to take tiny time steps.
- Check for accidental very small capacitors (e.g., `1p` when you meant `1u`) or very large resistors.
- A 1 pF capacitor with a 1 GΩ resistor has τ = 1 second — combined with a 1 ns signal, the solver needs both to be accurate simultaneously, requiring millions of steps.

---

## Category 4: Log Interpretation

Use `get_simulation_summary` to retrieve the simulation log.

**Serious warnings that must be addressed:**

| Message | Meaning | Fix |
|-|-|-|
| "Less than 2 connections at node X" | Node X is floating (only one connection) | Add ground resistor or fix wiring |
| "Bandwidth limited to Y Hz" | Op-amp GBW is lower than the simulation frequency | Use a faster op-amp model or reduce analysis frequency |
| "MOSFET M1 in linear region" | MOSFET is in triode, not saturation | Adjust bias (increase VDS) |
| "Can't find .model for X" | Missing device model | Add .lib or .model statement |
| "Singular matrix" | Floating node or voltage source loop | See Category 1 |

**Informational messages (usually safe to ignore):**

- "Curly braces in model" — LTspice model file format issue, simulation may still work
- "Warning: Vbe=..." for BJTs in cutoff/saturation — expected at circuit extremes

---

## Quick Fix Checklist (priority order)

Work through these in order — most simulation failures are resolved by step 1–4:

1. **Ground node?** Every circuit must have node 0.
2. **Floating nodes?** Every node needs a DC path to ground.
3. **Voltage source loops?** Add 1 mΩ series resistance.
4. **SPICE notation?** Use `1Meg` not `1M`. No spaces in values.
5. **AC source keyword?** `V1 in 0 AC 1` not `V1 in 0 1`.
6. **Relax reltol?** Try `.options reltol=0.003` first.
7. **Ramp sources?** Replace step edges with 1 ns rise/fall times.

If none of the above resolve the issue, use `read_netlist` to share the full netlist for detailed review.
"""

    return types.GetPromptResult(
        description=f"Simulation debugging workflow for: {problem_description}",
        messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=prompt_text))],
    )
