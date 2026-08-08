import streamlit as st
import pandas as pd
import joblib

from ultralytics import YOLO
from PIL import Image
import tempfile

import plotly.graph_objects as go
import plotly.express as px
import os
# -------------------------------
# # -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="AI Asset Intelligence Platform",
    page_icon="🏭",
    layout="wide"
)

# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("🏭 Asset Intelligence")


st.sidebar.success("🟢 System Status : Online")

st.sidebar.markdown("---")

st.sidebar.write("### Modules")

st.sidebar.write("✅ RUL Prediction")
st.sidebar.write("✅ Corrosion Detection")
st.sidebar.write("✅ Sensor Monitoring")
st.sidebar.write("✅ AI Decision Engine")
st.sidebar.write("✅ Priority Ranking")

st.sidebar.markdown("---")

st.sidebar.info("Prototype Version 1.0")

# =====================================================
# Main Title
# =====================================================

st.title("🏭 AI-Powered Asset Intelligence Platform")

st.caption(
    "An Explainable AI platform that combines Computer Vision, "
    "Sensor Data, Maintenance History, Engineer Notes and Weather "
    "to prioritize industrial assets for predictive maintenance."
)

st.markdown("---")

st.info("""
### 🚀 Intelligent Maintenance Decision Support System

This platform combines:

📷 Computer Vision

🌡 IoT Sensor Data

📋 Maintenance History

👨‍🔧 Engineer Notes

🌦 Weather Data

to prioritize industrial assets using Explainable AI.
""")


# -------------------------------
# Load Models
# -------------------------------
if os.path.exists("models/rul_model.pkl"):
    model = joblib.load("models/rul_model.pkl")
else:
    model = None

yolo_model = YOLO("models/best.pt")


# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv(
    "data/CMaps/train_FD001.txt",
    sep=r"\s+",
    header=None
)

df.drop(
    columns=[26, 27],
    inplace=True,
    errors="ignore"
)

# Column Names
cols = [
    "engine_no",
    "cycle",
    "op1",
    "op2",
    "op3"
]

for i in range(1, 22):
    cols.append(f"s{i}")

df.columns = cols


# -------------------------------
# Prepare Latest Engine Data
# -------------------------------

latest = df.groupby("engine_no").last().reset_index()

sensor_cols = [
    f"s{i}" for i in range(1, 22)
]

X = latest[sensor_cols]

if model is not None:
    latest["Predicted_RUL"] = model.predict(X)
else:
    latest["Predicted_RUL"] = 100


# -------------------------------
# Health Score
# -------------------------------

def health_score(rul):

    return round(
        max(0, min(100, (rul / 150) * 100)),
        2
    )


latest["Health"] = latest["Predicted_RUL"].apply(
    health_score
)



# -------------------------------
# Asset Selection
# -------------------------------

engine = st.selectbox(
    "Select Asset",
    latest["engine_no"],
    format_func=lambda x: f"Engine-{x}"
)


row = latest[
    latest.engine_no == engine
].iloc[0]


health = row["Health"]

rul = row["Predicted_RUL"]



# -------------------------------
# Risk Level (RUL Based)
# -------------------------------

if health >= 70:

    risk = "🟢 Low"
    recommendation = (
        "No maintenance required. Continue monitoring."
    )


elif health >= 40:

    risk = "🟠 Medium"
    recommendation = (
        "Schedule maintenance in upcoming cycles."
    )


else:

    risk = "🔴 High"
    recommendation = (
        "Immediate inspection and maintenance required."
    )

st.markdown("""
<style>
[data-testid="metric-container"] {
    background-color: #1e293b;
    border: 2px solid #334155;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Dashboard
# -------------------------------

col1, col2, col3 = st.columns(3)


col1.metric(
    "Health Score",
    f"{health}%"
)


col2.metric(
    "Predicted RUL",
    f"{round(rul)} Cycles"
)


col3.metric(
    "Risk Level",
    risk
)

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health,
    title={"text": "Health Score"},
    gauge={
        "axis": {"range": [0, 100]},
        "steps": [
            {"range": [0, 40], "color": "red"},
            {"range": [40, 70], "color": "orange"},
            {"range": [70, 100], "color": "green"}
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)


st.markdown("---")

if "corrosion_status" not in st.session_state:
    st.session_state.corrosion_status = "Not Checked"

corrosion_status = st.session_state.corrosion_status

# =====================================================
# Asset Profile
# =====================================================

st.markdown("---")
st.subheader("📋 Unified Asset Profile")

asset_id = f"Pipeline-{engine}"

last_inspection = "07 Aug 2026"

overall_status = (
    "🔴 Critical"
    if health < 40
    else "🟠 Warning"
    if health < 70
    else "🟢 Healthy"
)

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
**Asset ID:** {asset_id}

**Overall Status:** {overall_status}

**Health Score:** {health:.2f}%

**Predicted RUL:** {round(rul)} Cycles
""")

with col2:
    st.info(f"""
**Risk Level:** {risk}

**Corrosion Status:** {corrosion_status}

**Last Inspection:** {last_inspection}

**Recommendation:** {recommendation}
""")


# =====================================================
# Sensor Data
# =====================================================

st.markdown("---")
st.subheader("📈 Live Sensor Data")

# Dummy Sensor Values (later can be connected to real sensors)
temperature = 95      # °C
pressure = 12.4       # bar
vibration = "High"
humidity = 84         # %

c1, c2, c3, c4 = st.columns(4)

c1.metric("🌡 Temperature", f"{temperature} °C")
c2.metric("⚙ Pressure", f"{pressure} bar")
c3.metric("📳 Vibration", vibration)
c4.metric("💧 Humidity", f"{humidity}%")

# Sensor Status
if temperature > 90:
    st.error("⚠ High temperature detected.")

if vibration == "High":
    st.warning("⚠ Abnormal vibration detected.")

if humidity > 80:
    st.info("💧 High humidity may accelerate corrosion.")

# =====================================================
# Sensor Trend Chart
# =====================================================

sensor_df = pd.DataFrame({
    "Time": ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"],
    "Temperature": [
        temperature - 8,
        temperature - 6,
        temperature - 4,
        temperature - 2,
        temperature - 1,
        temperature
    ]
})

fig = px.line(
    sensor_df,
    x="Time",
    y="Temperature",
    title="🌡 Temperature Trend",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Maintenance History
# =====================================================

st.markdown("---")
st.subheader("📋 Maintenance History")

history = pd.DataFrame({
    "Date": [
        "12 Jul 2026",
        "05 Apr 2026",
        "18 Jan 2026"
    ],
    "Action": [
        "Routine Inspection",
        "Valve Replacement",
        "Rust Cleaning"
    ],
    "Engineer": [
        "Rahul",
        "Aman",
        "Priya"
    ],
    "Status": [
        "Completed",
        "Completed",
        "Completed"
    ]
})

st.dataframe(history, use_container_width=True)

# =====================================================
# Engineer Notes
# =====================================================

st.markdown("---")
st.subheader("📝 Engineer Notes")

st.info("""
**Latest Inspection Notes**

• Minor corrosion observed near the pipe joint.

• Vibration level has increased over the last 2 weeks.

• Temperature is higher than the normal operating range.

• Recommend detailed inspection during the next maintenance window.
""")

# =====================================================
# Weather & Operating Conditions
# =====================================================

st.markdown("---")
st.subheader("🌦 Weather & Operating Conditions")

w1, w2, w3 = st.columns(3)

w1.metric("🌡 Ambient Temp", "41°C")
w2.metric("💧 Humidity", "86%")
w3.metric("🌬 Wind Speed", "12 km/h")

st.warning("""
**Operating Environment**

• High ambient temperature

• High humidity may accelerate corrosion

• Outdoor pipeline exposed to harsh weather conditions
""")



# Calculate Failure Risk

failure_risk = 100 - health

if temperature > 90:
    failure_risk += 5

if vibration == "High":
    failure_risk += 5

failure_risk = min(100, round(failure_risk))


# =====================================================
# =====================================================
# AI Decision Engine
# =====================================================

st.markdown("---")
st.subheader("🧠 AI Decision Engine")

# -------------------------------
# Calculate AI Risk Score
# -------------------------------

score = 0
reasons = []

# Health Score
if health < 40:
    score += 30
    reasons.append("✔ Health Score is critically low.")
elif health < 70:
    score += 15
    reasons.append("✔ Health Score is below normal.")

# Corrosion Detection
if corrosion_status == "🔴 Corrosion Detected":
    score += 25
    reasons.append("✔ Corrosion detected from uploaded image.")

# Temperature
if temperature > 90:
    score += 20
    reasons.append("✔ High operating temperature detected.")
elif temperature > 75:
    score += 10
    reasons.append("✔ Temperature is slightly above normal.")

# Vibration
if vibration == "High":
    score += 15
    reasons.append("✔ High vibration detected.")
elif vibration == "Medium":
    score += 8
    reasons.append("✔ Moderate vibration detected.")

# Humidity
if humidity > 80:
    score += 10
    reasons.append("✔ High humidity may accelerate corrosion.")
elif humidity > 60:
    score += 5
    reasons.append("✔ Humidity is moderately high.")

# Maintenance History
score += 5
reasons.append("✔ Maintenance history indicates repeated repairs.")

# Final Failure Risk
failure_risk = min(score, 100)

# -------------------------------
# Display Failure Risk
# -------------------------------

st.metric("🚨 Failure Risk", f"{failure_risk}%")

risk_data = pd.DataFrame({
    "Category": ["Failure Risk", "Safe"],
    "Value": [failure_risk, 100 - failure_risk]
})

fig = px.pie(
    risk_data,
    values="Value",
    names="Category",
    title="Failure Risk Distribution",
    hole=0.5
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# AI Explanation
# -------------------------------

st.write("### 📌 Why did AI classify this asset?")

for reason in reasons:
    st.write(reason)

# -------------------------------
# AI Risk Score
# -------------------------------

st.markdown("### 🧠 AI Decision Score")

st.progress(failure_risk / 100)

st.success(f"Final AI Risk Score : {failure_risk}/100")

# -------------------------------
# AI Decision
# -------------------------------

if failure_risk >= 80:

    st.error("🚨 Immediate maintenance required within 24 hours.")

elif failure_risk >= 60:

    st.warning("⚠️ Schedule inspection as soon as possible.")

else:

    st.success("✅ Asset is healthy. Continue monitoring.")

# =====================================================
# AI Decision Explanation
# =====================================================

st.markdown("### 🧠 AI Decision Explanation")

score = 0

if corrosion_status == "🔴 Corrosion Detected":
    score += 30
    st.write("🔴 Corrosion detected from uploaded image (+30)")

if health < 40:
    score += 25
    st.write("🟠 Health Score is critically low (+25)")

if temperature > 90:
    score += 15
    st.write("🌡 High operating temperature (+15)")

if vibration == "High":
    score += 15
    st.write("📳 High vibration detected (+15)")

if humidity > 80:
    score += 10
    st.write("💧 High humidity may accelerate corrosion (+10)")

score += 5
st.write("📋 Maintenance history indicates repeated issues (+5)")

st.progress(min(score, 100) / 100)

st.success(f"Final AI Risk Score: {min(score,100)}/100")

if score >= 70:
    st.error("🚨 AI Decision: Immediate maintenance required within 24 hours.")
elif score >= 40:
    st.warning("⚠️ AI Decision: Schedule inspection soon.")
else:
    st.success("✅ AI Decision: Asset is healthy. Continue monitoring.")

if score >= 80:
    action = "Repair within 24 hrs"
elif score >= 60:
    action = "Inspect Today"
elif score >= 40:
    action = "Monitor Weekly"
else:
    action = "Routine Inspection"

st.info(f"""
### 🤖 Final AI Decision Summary

**Asset:** {asset_id}

**Risk Score:** {min(score,100)}/100

**Priority:** Rank #1

**Recommended Action:** {action}

**Reason:** AI analyzed image, sensor data, maintenance history, engineer notes and weather conditions before generating this decision.
""") 


# =====================================================
# Asset Priority Ranking
# =====================================================
st.markdown("---")
st.subheader("🚨 Asset Priority Ranking")

# Dynamic Risk
if failure_risk >= 80:
    risk_label = "🔴 Critical"
    action = "Repair within 24 hrs"
elif failure_risk >= 60:
    risk_label = "🟠 High"
    action = "Inspect Today"
elif failure_risk >= 40:
    risk_label = "🟡 Medium"
    action = "Monitor Weekly"
else:
    risk_label = "🟢 Low"
    action = "Routine Inspection"

priority = pd.DataFrame({
    "Rank": ["🥇 1", "🥈 2", "🥉 3", "4"],
    "Asset": [
        asset_id,
        "Valve-18",
        "Pump-07",
        "Tank-03"
    ],
    "Risk": [
        risk_label,
        "🟠 High",
        "🟡 Medium",
        "🟢 Low"
    ],
    "Recommended Action": [
        action,
        "Inspect Today",
        "Monitor Weekly",
        "Routine Inspection"
    ]
})

st.dataframe(priority, use_container_width=True)

# =====================================================
# Asset Risk Comparison Chart
# =====================================================

asset_df = pd.DataFrame({
    "Asset": ["Pipeline-1", "Valve-18", "Pump-07", "Tank-03"],
    "Risk Score": [
        failure_risk,
        70,
        45,
        20
    ]
})

fig = px.bar(
    asset_df,
    x="Asset",
    y="Risk Score",
    title="📊 Asset Risk Comparison",
    text="Risk Score"
)

fig.update_traces(textposition="outside")

fig.update_layout(
    yaxis_range=[0, 100]
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Maintenance Recommendation")

if failure_risk >= 80:
    st.error("🚨 Immediate inspection and maintenance required.")

elif failure_risk >= 60:
    st.warning("⚠️ Schedule inspection as soon as possible.")

else:
    st.success("✅ Continue monitoring the asset.")


# =====================================================
# Corrosion Detection
# =====================================================

st.markdown("---")
st.subheader("🔍 Corrosion Detection")

uploaded_file = st.file_uploader(
    "Upload Asset Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=450
    )

    if st.button("🔍 Predict Corrosion"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:

            image.save(tmp.name)

            results = yolo_model.predict(
                tmp.name,
                conf=0.10
            )

        result = results[0]

        annotated_image = result.plot()

        st.subheader("Detection Result")

        st.image(
            annotated_image,
            use_container_width=True
        )

        # -----------------------------
        # Corrosion Status
        # -----------------------------

        if len(result.boxes) > 0:

            st.session_state.corrosion_status = "🔴 Corrosion Detected"
            corrosion_status = st.session_state.corrosion_status

            st.error("⚠ Corrosion Detected")

            st.write(f"Detected Regions : {len(result.boxes)}")

            st.write("### Detection Confidence")

            for i, box in enumerate(result.boxes):

                confidence = float(box.conf[0])

                st.write(
                    f"Region {i+1} : {confidence*100:.2f}%"
                )

            failure_risk = min(
                100,
                round(
                    (100-health)+20
                )
            )

            st.error(
                f"""
### 🚨 AI Decision

Failure Risk : **{failure_risk}%**

Immediate maintenance required.

Reason:

✔ Corrosion detected

✔ Low Health Score

✔ Low Remaining Useful Life

✔ High Temperature

✔ High Vibration
"""
            )
            

        else:

            st.session_state.corrosion_status = "🟢 No Corrosion Detected"
            corrosion_status = st.session_state.corrosion_status

            st.success("✅ No Corrosion Detected")

            failure_risk = min(
                100,
                round(
                    (100-health)
                )
            )

            st.success(
                f"""
### 🤖 AI Decision

Failure Risk : **{failure_risk}%**

No corrosion found.

Continue monitoring.
"""
            )
            st.rerun()


st.markdown("---")

st.caption("""
🏭 AI Asset Intelligence Platform

Developed by Team AurixGroup

Image + Sensor + Maintenance + Weather → AI Decision
""")