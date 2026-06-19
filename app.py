from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("titanic_model.pkl")
data = pd.read_csv("train.csv")


def page_style():
    return """
    <style>
        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            background:
                linear-gradient(rgba(14, 35, 52, 0.72), rgba(14, 35, 52, 0.78)),
                repeating-linear-gradient(
                    90deg,
                    #6b4427 0px,
                    #6b4427 26px,
                    #5a351f 26px,
                    #5a351f 52px
                );
            font-family: Georgia, 'Times New Roman', serif;
            color: #24170e;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px;
        }

        .ticket {
            width: 900px;
            background: #efe0bd;
            border: 6px solid #122b40;
            outline: 3px solid #b08a3c;
            padding: 30px;
            box-shadow: 0 25px 70px rgba(0,0,0,0.45);
            position: relative;
        }

        .ticket::before {
            content: "";
            position: absolute;
            inset: 14px;
            border: 2px solid #7d5a24;
            pointer-events: none;
        }

        .top {
            display: grid;
            grid-template-columns: 160px 1fr 160px;
            align-items: center;
            border-bottom: 4px double #122b40;
            padding-bottom: 18px;
            margin-bottom: 24px;
            position: relative;
            z-index: 1;
        }

        .badge {
            border: 2px solid #7d5a24;
            padding: 10px;
            text-align: center;
            font-size: 12px;
            letter-spacing: 2px;
            color: #5d3e12;
            background: #f7ebcc;
        }

        h1 {
            margin: 0;
            text-align: center;
            font-size: 34px;
            letter-spacing: 4px;
            color: #122b40;
        }

        .subtitle {
            text-align: center;
            margin-top: 7px;
            font-size: 14px;
            color: #5d3e12;
            letter-spacing: 1px;
        }

        .form-grid, .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px 22px;
            position: relative;
            z-index: 1;
        }

        label {
            display: block;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: bold;
            color: #122b40;
            margin-bottom: 7px;
        }

        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #7d5a24;
            background: #fff4d5;
            font-family: Georgia, serif;
            font-size: 15px;
            color: #24170e;
            border-radius: 0;
        }

        .full { grid-column: 1 / 3; }

        .gender-choice {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        .gender-card input {
            display: none;
        }

        .gender-card label {
            display: block;
            border: 3px solid #7d5a24;
            background: #f7ebcc;
            padding: 14px;
            text-align: center;
            cursor: pointer;
            min-height: 190px;
            transition: 0.2s;
        }

        .gender-card input:checked + label {
            border-color: #122b40;
            background: #d7c08b;
            box-shadow: inset 0 0 0 3px #b08a3c;
        }

        .person-name {
            margin-top: 8px;
            font-size: 14px;
            letter-spacing: 2px;
            color: #122b40;
        }

        svg {
            width: 90px;
            height: 130px;
        }

        button {
            width: 100%;
            margin-top: 8px;
            padding: 15px;
            border: 3px solid #b08a3c;
            background: #122b40;
            color: #fff4d5;
            font-family: Georgia, serif;
            font-size: 15px;
            font-weight: bold;
            letter-spacing: 3px;
            cursor: pointer;
            border-radius: 0;
        }

        button:hover {
            background: #1e4564;
        }

        .panel {
            background: #f7ebcc;
            border: 3px solid #7d5a24;
            padding: 18px;
        }

        .stamp {
            display: inline-block;
            border: 4px double #8b1e1e;
            color: #8b1e1e;
            padding: 14px 28px;
            font-size: 28px;
            font-weight: bold;
            transform: rotate(-4deg);
            letter-spacing: 4px;
            margin: 12px 0;
        }

        .prob {
            font-size: 42px;
            font-weight: bold;
            color: #122b40;
        }

        .bar {
            height: 18px;
            border: 2px solid #122b40;
            overflow: hidden;
            background: #efe0bd;
            margin-top: 12px;
        }

        .fill {
            height: 100%;
            background: #122b40;
        }

        .factor {
            margin: 8px 0;
            padding: 8px 10px;
            border-left: 5px solid #122b40;
            background: #fff4d5;
        }

        .back {
            display: inline-block;
            margin-top: 24px;
            color: #122b40;
            font-weight: bold;
            text-decoration: none;
            border-bottom: 2px solid #122b40;
        }

        @media (max-width: 800px) {
            .ticket { width: 100%; }
            .top { grid-template-columns: 1fr; gap: 10px; }
            .form-grid, .result-grid { grid-template-columns: 1fr; }
            .full { grid-column: 1; }
            .gender-choice { grid-template-columns: 1fr; }
            h1 { font-size: 26px; }
        }
    </style>
    """


def female_svg():
    return """
    <svg viewBox="0 0 120 170">
        <circle cx="60" cy="28" r="18" fill="#f1c8a8" stroke="#122b40" stroke-width="3"/>
        <path d="M42 25 C38 5, 82 5, 78 25 C75 14, 45 14, 42 25Z" fill="#2b1a12"/>
        <path d="M37 48 C50 60, 70 60, 83 48 L92 118 L28 118 Z" fill="#2f4f6f" stroke="#122b40" stroke-width="3"/>
        <path d="M45 55 L60 90 L75 55" fill="none" stroke="#fff4d5" stroke-width="4"/>
        <path d="M28 118 L92 118 L103 158 L17 158 Z" fill="#7d2f2f" stroke="#122b40" stroke-width="3"/>
        <line x1="40" y1="158" x2="40" y2="168" stroke="#122b40" stroke-width="4"/>
        <line x1="80" y1="158" x2="80" y2="168" stroke="#122b40" stroke-width="4"/>
        <line x1="28" y1="72" x2="10" y2="105" stroke="#122b40" stroke-width="4"/>
        <line x1="92" y1="72" x2="110" y2="105" stroke="#122b40" stroke-width="4"/>
    </svg>
    """


def male_svg():
    return """
    <svg viewBox="0 0 120 170">
        <circle cx="60" cy="30" r="18" fill="#f1c8a8" stroke="#122b40" stroke-width="3"/>
        <path d="M40 22 C45 5, 76 5, 80 22 L74 15 L46 15 Z" fill="#2b1a12"/>
        <path d="M35 52 L85 52 L92 112 L28 112 Z" fill="#122b40" stroke="#122b40" stroke-width="3"/>
        <path d="M50 52 L60 88 L70 52" fill="#fff4d5"/>
        <path d="M57 58 L63 58 L66 95 L54 95 Z" fill="#7d2f2f"/>
        <rect x="32" y="112" width="22" height="46" fill="#2f4f6f" stroke="#122b40" stroke-width="3"/>
        <rect x="66" y="112" width="22" height="46" fill="#2f4f6f" stroke="#122b40" stroke-width="3"/>
        <line x1="35" y1="62" x2="15" y2="108" stroke="#122b40" stroke-width="4"/>
        <line x1="85" y1="62" x2="105" y2="108" stroke="#122b40" stroke-width="4"/>
    </svg>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
    <html>
    <head>{page_style()}<title>Titanic Predictor</title></head>
    <body>
        <div class="ticket">
            <div class="top">
                <div class="badge">WHITE STAR LINE</div>
                <div>
                    <h1>PASSENGER REPORT</h1>
                    <div class="subtitle">Titanic survival prediction archive</div>
                </div>
                <div class="badge">APRIL 1912</div>
            </div>

            <form action="/predict" method="post">
                <div class="form-grid">
                    <div>
                        <label>Passenger Class</label>
                        <select name="pclass">
                            <option value="1">First Class</option>
                            <option value="2">Second Class</option>
                            <option value="3">Third Class</option>
                        </select>
                    </div>

                    <div>
                        <label>Age</label>
                        <input type="number" name="age" value="25">
                    </div>

                    <div class="full">
                        <label>Passenger Type</label>
                        <div class="gender-choice">
                            <div class="gender-card">
                                <input type="radio" id="female" name="sex" value="1" checked>
                                <label for="female">
                                    {female_svg()}
                                    <div class="person-name">LADY PASSENGER</div>
                                </label>
                            </div>

                            <div class="gender-card">
                                <input type="radio" id="male" name="sex" value="0">
                                <label for="male">
                                    {male_svg()}
                                    <div class="person-name">GENTLEMAN PASSENGER</div>
                                </label>
                            </div>
                        </div>
                    </div>

                    <div>
                        <label>Ticket Fare</label>
                        <input type="number" name="fare" value="50">
                    </div>

                    <div>
                        <label>Siblings / Spouse</label>
                        <input type="number" name="sibsp" value="0">
                    </div>

                    <div>
                        <label>Parents / Children</label>
                        <input type="number" name="parch" value="0">
                    </div>

                    <div class="full">
                        <button type="submit">CHECK PASSENGER</button>
                    </div>
                </div>
            </form>
        </div>
    </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
def predict(
    pclass: int = Form(...),
    sex: int = Form(...),
    age: float = Form(...),
    sibsp: int = Form(...),
    parch: int = Form(...),
    fare: float = Form(...)
):
    person = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare
    }])

    prediction = model.predict(person)[0]
    probability = model.predict_proba(person)[0][1] * 100

    gender_text = "Female" if sex == 1 else "Male"
    class_text = f"{pclass} Class"
    result = "SURVIVED" if prediction == 1 else "LOST AT SEA"

    similar = data[
        (data["Pclass"] == pclass) &
        (data["Sex"] == ("female" if sex == 1 else "male"))
    ]
    similar_rate = similar["Survived"].mean() * 100

    factors = []
    if sex == 1:
        factors.append("Female passengers had a higher survival rate.")
    else:
        factors.append("Male passengers had a lower survival rate.")

    if pclass == 1:
        factors.append("First Class passengers had better survival chances.")
    elif pclass == 3:
        factors.append("Third Class passengers had lower survival chances.")

    if fare > data["Fare"].median():
        factors.append("Higher ticket fare may indicate better access and class position.")
    else:
        factors.append("Lower ticket fare may be linked with lower survival chances.")

    if age < 16:
        factors.append("Children often had better evacuation priority.")

    factor_html = "".join([f"<div class='factor'>{f}</div>" for f in factors])

    story = f"""
    {gender_text} passenger, {age:.0f} years old, travelling in {class_text}.
    Ticket fare: ${fare:.2f}. The model compares this profile with historical Titanic passenger data.
    """

    return f"""
    <html>
    <head>{page_style()}<title>Result</title></head>
    <body>
        <div class="ticket">
            <div class="top">
                <div class="badge">ARCHIVE RESULT</div>
                <div>
                    <h1>PASSENGER ANALYSIS</h1>
                    <div class="subtitle">Random Forest model prediction</div>
                </div>
                <div class="badge">CONFIDENTIAL</div>
            </div>

            <div class="result-grid">
                <div class="panel">
                    <p>Passenger Status</p>
                    <div class="stamp">{result}</div>

                    <p>Estimated Survival Probability</p>
                    <div class="prob">{probability:.2f}%</div>

                    <div class="bar">
                        <div class="fill" style="width: {probability:.2f}%"></div>
                    </div>
                </div>

                <div class="panel">
                    <h3>Passenger Card</h3>
                    <p><b>Class:</b> {class_text}</p>
                    <p><b>Gender:</b> {gender_text}</p>
                    <p><b>Age:</b> {age:.0f}</p>
                    <p><b>Fare:</b> ${fare:.2f}</p>
                    <p><b>Family aboard:</b> {sibsp + parch}</p>
                </div>

                <div class="panel">
                    <h3>Model Explanation</h3>
                    {factor_html}
                </div>

                <div class="panel">
                    <h3>Historical Comparison</h3>
                    <p>Similar passengers survived in approximately:</p>
                    <div class="prob">{similar_rate:.2f}%</div>
                </div>

                <div class="panel full">
                    <h3>Passenger Story</h3>
                    <p>{story}</p>
                    <a class="back" href="/">Check another passenger</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """