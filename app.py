 
import numpy as np
import streamlit as st
import cantera as ct
import plotly.graph_objects as go


# -----------------------------
# Cantera phases
# -----------------------------
@st.cache_resource
def get_gas():
    # GRI3.0 includes CH4, CO2, H2O, CO, H2 etc.
    return ct.Solution("gri30.yaml")


@st.cache_resource
def get_graphite():
    # Cantera typically ships graphite.yaml (depending on version)
    for name in ["graphite.yaml", "graphite.cti", "graphite.xml"]:
        try:
            return ct.Solution(name)
        except Exception:
            pass
    return None


# -----------------------------
# Composition helpers
# -----------------------------
def to_dry(y_dict):
    keys = ["H2", "CO", "CO2", "CH4"]
    s = sum(max(y_dict.get(k, 0.0), 0.0) for k in keys)
    if s <= 0:
        return {k: 0.0 for k in keys}
    return {k: 100.0 * y_dict.get(k, 0.0) / s for k in keys}


def to_wet(y_dict):
    keys = ["H2O", "H2", "CO", "CO2", "CH4"]
    s = sum(max(y_dict.get(k, 0.0), 0.0) for k in keys)
    if s <= 0:
        return {k: 0.0 for k in keys}
    return {k: 100.0 * y_dict.get(k, 0.0) / s for k in keys}


# -----------------------------
# Thermo: Kp from standard Gibbs
# -----------------------------
def reaction_Kp(T, gas, graphite, nu_gas, nu_graphite=None):
    """
    Compute Kp for a reaction from standard-state Gibbs energies (NASA polynomials inside Cantera).
    Standard state: 1 bar.
    """
    gas.TP = T, ct.one_atm
    g0_RT = gas.standard_gibbs_RT

    dg0_RT = 0.0
    for sp, nu in nu_gas.items():
        dg0_RT += nu * g0_RT[gas.species_index(sp)]

    if nu_graphite and graphite is not None:
        graphite.TP = T, ct.one_atm
        g0g_RT = graphite.standard_gibbs_RT
        for sp, nu in nu_graphite.items():
            dg0_RT += nu * g0g_RT[graphite.species_index(sp)]

    return np.exp(-dg0_RT)  # Kp = exp(-ΔG°/RT)


def carbon_risk(T, P_bar, y_gas_eq, gas, graphite):
    """
    Carbon risk by two classic reactions:
      (A) CH4 -> C(gr) + 2H2      Kp = pH2^2 / pCH4
      (B) 2CO -> C(gr) + CO2      Kp = pCO2 / pCO^2
    margin = ln(Q) - ln(K). negative => risk (ΔG < 0)
    """
    pCH4 = max(y_gas_eq.get("CH4", 0.0), 0.0) * P_bar
    pH2  = max(y_gas_eq.get("H2",  0.0), 0.0) * P_bar
    pCO  = max(y_gas_eq.get("CO",  0.0), 0.0) * P_bar
    pCO2 = max(y_gas_eq.get("CO2", 0.0), 0.0) * P_bar

    eps = 1e-30
    Q1 = (pH2**2) / (pCH4 + eps)
    Q2 = (pCO2) / (pCO**2 + eps)

    if graphite is None:
        return {
            "allow_any": False,
            "allow_ch4": False,
            "allow_bou": False,
            "margin_ch4": np.nan,
            "margin_bou": np.nan,
            "note": "graphite phase not found in Cantera install",
        }

    K1 = reaction_Kp(
        T, gas, graphite,
        nu_gas={"CH4": -1, "H2": +2},
        nu_graphite={"C(gr)": +1},
    )
    K2 = reaction_Kp(
        T, gas, graphite,
        nu_gas={"CO": -2, "CO2": +1},
        nu_graphite={"C(gr)": +1},
    )

    m1 = np.log(max(Q1, eps)) - np.log(max(K1, eps))
    m2 = np.log(max(Q2, eps)) - np.log(max(K2, eps))

    allow_ch4 = (Q1 < K1) and (pCH4 > 1e-12)
    allow_bou = (Q2 < K2) and (pCO  > 1e-12)

    return {
        "allow_any": allow_ch4 or allow_bou,
        "allow_ch4": allow_ch4,
        "allow_bou": allow_bou,
        "margin_ch4": m1,
        "margin_bou": m2,
        "note": "",
    }


# -----------------------------
# Plotly helpers (mobile-safe)
# -----------------------------
def find_regions(x, flag01):
    """Convert 0/1 flags on x-grid into contiguous [x0,x1] regions."""
    regions = []
    start = None
    for i in range(len(x)):
        on = (flag01[i] == 1)
        last = (i == len(x) - 1)
        if on and start is None:
            start = x[i]
        if start is not None and ((not on) or last):
            end = x[i] if (on and last) else x[i - 1]
            if end > start:
                regions.append((start, end))
            start = None
    return regions


def apply_mobile_layout(fig, title_text, y_title, y_is_percent=True):
    """
    Fix axis ticks, font sizes, margins and legend so mobile doesn't overlap.
    """
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16), x=0.5, xanchor="center"),
        margin=dict(l=60, r=20, t=55, b=90),  # bottom margin big to avoid overlap with legend/xlabel
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.30,           # push legend below plot area
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
    )

    fig.update_xaxes(
        title=dict(text="Temperature (°C)", standoff=18),
        tickfont=dict(size=10),
        tickmode="linear",
        tick0=400,
        dtick=100,
        fixedrange=True
    )

    fig.update_yaxes(
        title=dict(text=y_title, standoff=10),
        tickfont=dict(size=10),
        fixedrange=True,
    )

    if y_is_percent:
        fig.update_yaxes(
            range=[0, 100],
            tickmode="linear",
            tick0=0,
            dtick=20,
        )

    return fig


def stacked_area(x, series_dict, title, ytitle, danger_regions):
    fig = go.Figure()
    # stack order: as given in dict
    for name, y in series_dict.items():
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", stackgroup="one", name=name
        ))

    # Danger shading
    for (x0, x1) in danger_regions:
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="rgba(255,0,0,0.12)",
            line_width=0,
            layer="below"
        )

    apply_mobile_layout(fig, title, ytitle, y_is_percent=True)
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Biogas Reformer Simulator", layout="wide")

st.markdown(
    "<h2 style='margin-bottom:0.2rem;'>Biogas Reformer Simulator (NASA thermo via Cantera)</h2>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Inputs")
    P_atm = st.slider("Pressure (atm)", 1.0, 2.0, 1.20, 0.01)
    S_C   = st.slider("S/C", 0.0, 6.0, 2.50, 0.05)
    xCH4  = st.slider("Biogas CH4 fraction (rest CO2)", 0.30, 0.90, 0.60, 0.01)
    npts  = st.slider("Temperature points", 21, 101, 41, 2)
    st.caption("Biogas basis: 1 mol = CH4 + CO2(rest). Steam = S/C × CH4 mol.")

gas = get_gas()
graphite = get_graphite()

T_C = np.linspace(400, 800, npts)
P_bar = P_atm * 1.01325

dry = {"H2": [], "CO": [], "CO2": [], "CH4": []}
wet = {"H2O": [], "H2": [], "CO": [], "CO2": [], "CH4": []}
flag = []
m_ch4 = []
m_bou = []

for Tc in T_C:
    T = Tc + 273.15

    nCH4 = xCH4
    nCO2 = 1.0 - xCH4
    nH2O = S_C * xCH4
    comp = {"CH4": nCH4, "CO2": nCO2, "H2O": nH2O}

    # Equilibrium in gas phase at TP (Gibbs minimization)
    gas.TPX = T, P_bar * 1e5, comp  # Pa
    gas.equilibrate("TP")

    y = {sp: gas.X[gas.species_index(sp)] for sp in ["H2O", "H2", "CO", "CO2", "CH4"]}

    d = to_dry(y)
    w = to_wet(y)

    for k in dry.keys():
        dry[k].append(d[k])
    for k in wet.keys():
        wet[k].append(w[k])

    c = carbon_risk(T, P_bar, y, gas, graphite)
    flag.append(1 if c["allow_any"] else 0)
    m_ch4.append(c["margin_ch4"])
    m_bou.append(c["margin_bou"])

danger_regions = find_regions(T_C, flag)
show_warn = any(flag)

colA, colB = st.columns([1, 1], gap="large")

with colA:
    fig1 = stacked_area(
        T_C,
        {"H2": dry["H2"], "CO": dry["CO"], "CO2": dry["CO2"], "CH4": dry["CH4"]},
        "Dry gas composition (normalized to 100%)",
        "Dry composition (%)",
        danger_regions
    )
    if show_warn:
        st.warning("Carbon deposition is thermodynamically allowed in the shaded temperature range (based on CH4 cracking / Boudouard).")
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    fig2 = stacked_area(
        T_C,
        {"H2O": wet["H2O"], "H2": wet["H2"], "CO": wet["CO"], "CO2": wet["CO2"], "CH4": wet["CH4"]},
        "Wet gas composition (including steam, normalized to 100%)",
        "Wet composition (%)",
        danger_regions
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Carbon deposition margin (ln(Q) - ln(K); negative = risk)")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=T_C, y=m_ch4, mode="lines", name="CH4 → C + 2H2"))
fig3.add_trace(go.Scatter(x=T_C, y=m_bou, mode="lines", name="2CO → C + CO2"))

for (x0, x1) in danger_regions:
    fig3.add_vrect(x0=x0, x1=x1, fillcolor="rgba(255,0,0,0.12)", line_width=0, layer="below")

fig3.add_hline(y=0.0, line_dash="dash", line_width=1)

apply_mobile_layout(
    fig3,
    "Carbon deposition margin (ln(Q) - ln(K); negative = risk)",
    "ln(Q) - ln(K)",
    y_is_percent=False
)

# For margin plot, set a nicer x ticks and a little extra bottom margin
fig3.update_layout(margin=dict(l=60, r=20, t=55, b=85))
fig3.update_xaxes(tickmode="linear", tick0=400, dtick=100)

st.plotly_chart(fig3, use_container_width=True)

if graphite is None:
    st.info("Note: graphite phase file was not found in your Cantera install, so Kp-based carbon warning may be disabled. Try upgrading Cantera or adding graphite.yaml.")
